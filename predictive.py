import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [35, 40, 45, 50, 55, 60, 65, 70, 78, 85]
}

df = pd.DataFrame(data)

print("----- PREDICTIVE ANALYSIS -----")
print("\nDataset:")
print(df)

# Input and output
X = df[["Hours"]]
y = df["Score"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully.")

# Predict test data
y_pred = model.predict(X_test)

print("\nActual Scores:")
print(y_test.values)

print("\nPredicted Scores:")
print(y_pred)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Predict score for new student
hours = [[7.5]]
predicted_score = model.predict(hours)

print("\nPredicted score for 7.5 hours of study:",
      round(predicted_score[0], 2))

# Display graph
plt.scatter(
    df["Hours"],
    df["Score"],
    color="blue",
    label="Actual Data"
)

plt.plot(
    df["Hours"],
    model.predict(df[["Hours"]]),
    color="red",
    label="Regression Line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Predictive Analysis using Linear Regression")
plt.legend()
plt.show()
# Updated in predictive-analysis branch
