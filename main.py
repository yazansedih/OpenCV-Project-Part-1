
# Yazan Al Sedih
# 12010059
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

r_input_image = cv2.imread("input_image.jpg", cv2.IMREAD_GRAYSCALE)

ID = "12010059"
random.seed(int(ID))
gamma = random.uniform(0.04, 25)
# gamma = random.random();

# Apply gamma correction to each pixel in the input image
s_output_image = np.uint8(cv2.pow(r_input_image / 255.0, gamma) * 255)

c = 255 / (255 ** gamma)
print("Gamma:", gamma)
print("Constant c:", c)

# Display images before and after modifying brightness
cv2.imshow("Before", r_input_image)
cv2.imshow("After", s_output_image)

# Calculate histograms
hist_input = cv2.calcHist([r_input_image], [0], None, [256], [0, 256])
hist_output = cv2.calcHist([s_output_image], [0], None, [256], [0, 256])

# Display histograms
plt.figure()
plt.title("Histograms")
plt.plot(hist_input, color='black', label='Before')
plt.plot(hist_output, color='red', label='After')
plt.xlabel('Intensity')
plt.ylabel('# of pixels')
plt.legend()
plt.show()

# Save images with ID
cv2.imwrite("input_"+ ID +".jpg", r_input_image)
cv2.imwrite("output_"+ ID +".jpg", s_output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
