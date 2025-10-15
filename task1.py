import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def balance_data(data):
    gamma_data = data[data['class']=='g']
    hadron_data = data[data['class']=='h']
    min_size = min(len(gamma_data), len(hadron_data))
    balanced_gamma = gamma_data.sample(n=min_size, random_state=42)
    balanced_hadron = hadron_data.sample(n=min_size, random_state=42)
    balanced_data = pd.concat([balanced_gamma, balanced_hadron])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_data

def training_data_set(data):
    train_size = int(0.7 * len(data))
    training_data = data.iloc[:train_size].reset_index(drop=True)
    return training_data

def validation_data_set(data):
    val_size = int(0.15 * len(data))
    validation_data = data.iloc[int(0.7 * len(data)):int(0.7 * len(data)) + val_size].reset_index(drop=True)
    return validation_data

def test_data_set(data):
    test_size = int(0.15 * len(data))
    test_data = data.iloc[int(0.85 * len(data)):].reset_index(drop=True)
    return test_data

def distance_calculation(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2))

# def K_Nearest_Neighbors_Manual(data, k, test_sample):
#     distances =[]
#     for data1 in data.values:
#         data_train = data1[:-1]
#         dist = distance_calculation(data_train, test_sample)
#         label = data1[-1]
#         distances.append((dist, label))
        
#     distances.sort(key=lambda x: x[0])
#     neighbors = distances[:k]
#     gamma_count = sum(1 for neighbor in neighbors if neighbor[1] == 'g')
#     hadron_count = k - gamma_count
#     if gamma_count > hadron_count:
#         return 'g'
#     return 'h'

def K_Nearest_Neighbors_Manual(data, k, test_sample):
    X_train = data.iloc[:, :-1].values.astype(float)
    y_train = data.iloc[:, -1].values
    test_sample = np.array(test_sample).astype(float)

    # Vectorized distance calculation (no loops)
    distances = np.sqrt(np.sum((X_train - test_sample) ** 2, axis=1))

    # Sort distances and pick k nearest
    k_idx = np.argsort(distances)[:k]
    k_labels = y_train[k_idx]

    # Majority vote
    gamma_count = np.sum(k_labels == 'g')
    hadron_count = k - gamma_count
    return 'g' if gamma_count > hadron_count else 'h'

def K_Nearest_Neighbors_Sklearn(data, k, test_sample):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x, y)
    prediction = model.predict([test_sample])
    return prediction[0]

def accuracy_manual(train_data, k, test_data):
    X_train = train_data.iloc[:, :-1].values.astype(float)
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values.astype(float)
    y_test = test_data.iloc[:, -1].values

    correct = 0
    for test_sample, true_label in zip(X_test, y_test):
        distances = np.sqrt(np.sum((X_train - test_sample)**2, axis=1))
        k_idx = np.argsort(distances)[:k]
        k_labels = y_train[k_idx]
        prediction = 'g' if np.sum(k_labels == 'g') > np.sum(k_labels == 'h') else 'h'
        if prediction == true_label:
            correct += 1
    return correct / len(y_test)


def accuracy_sklearn(train_data, k, test_data):
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy


def plot_accuracies(k_values, manual_accuracies, sklearn_accuracies):
    plt.plot(k_values, manual_accuracies, label='Manual KNN', marker='o')
    plt.plot(k_values, sklearn_accuracies, label='Sklearn KNN', marker='x')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy Comparison')
    plt.legend()
    plt.grid()
    plt.show()

# Load and prepare data
data = load_data("telescope_data.csv")
balanced_data = balance_data(data)

# Split data FIRST (before scaling)
train = training_data_set(balanced_data)
val = validation_data_set(balanced_data)
test = test_data_set(balanced_data)

print("Validation class distribution:")
print(val['class'].value_counts())

# NOW scale: fit on training data only, transform all sets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit scaler on training data only
scaler.fit(train.iloc[:, :-1])

# Transform all sets using the same scaler
train.iloc[:, :-1] = scaler.transform(train.iloc[:, :-1])
val.iloc[:, :-1] = scaler.transform(val.iloc[:, :-1])
test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])

# Try a range of k values
k_values = range(1, 11)
manual_accuracies = []
sklearn_accuracies = []

for k in k_values:
    acc_manual = accuracy_manual(train, k, val)
    acc_sklearn = accuracy_sklearn(train, k, val)
    manual_accuracies.append(acc_manual)
    sklearn_accuracies.append(acc_sklearn)
    print(f"k={k}  →  Manual: {acc_manual:.3f},  Sklearn: {acc_sklearn:.3f}")


# Plot both accuracy curves
plot_accuracies(k_values, manual_accuracies, sklearn_accuracies)

# Final evaluation on test set with best k
best_k = k_values[np.argmax(sklearn_accuracies)]
print(f"\nTest class distribution:")
print(test['class'].value_counts())
final_accuracy = accuracy_sklearn(train, best_k, test)
print(f"Final accuracy on test set with k={best_k}: {final_accuracy:.3f}")