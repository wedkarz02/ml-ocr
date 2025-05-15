import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog


def load_data(num_samples=1000, image_size=(28, 28)):
    from sklearn.datasets import load_digits
    digits = load_digits()
    images = [cv2.resize(img, image_size) for img in digits.images]
    labels = digits.target
    return np.array(images), np.array(labels)


def preprocess(images):
    processed = []
    for img in images:
        gray = img.astype(np.uint8)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        processed.append(norm)
    return np.array(processed)


def extract_features(images):
    features = []
    for img in images:
        hog_feat = hog(img, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_feat)
    return np.array(features)


def plot_learning_curve(train_scores, test_scores, label):
    plt.plot(train_scores, label=f"{label} - train")
    plt.plot(test_scores, label=f"{label} - test")
    plt.xlabel("Epochs or Iterations")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"[{name}] Test Accuracy: {acc:.4f}")
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    return acc


def main():
    images, labels = load_data()
    images = preprocess(images)
    features = extract_features(images)

    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "SVM (RBF Kernel)": SVC(kernel='rbf'),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        acc = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results[name] = acc

    best_model = max(results, key=results.get)
    print(
        f"\n✅ Best model: {best_model} with accuracy {results[best_model]:.4f}")

    ks = range(1, 10)
    accs = []
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, model.predict(X_test)))
    plt.plot(ks, accs)
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("KNN - Impact of Parameter k")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
