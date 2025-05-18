// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-span %t

// Mock std::vector and std::array for testing
using size_t = unsigned long;
using ptrdiff_t = long;

namespace std {
template <typename T>
struct remove_cv { using type = T; };
template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

template <typename Iter>
class reverse_iterator {};

template <typename T>
class vector {
public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;
  const_iterator begin() const { return nullptr; }
  const_iterator end() const { return nullptr; }
};

template <typename T, size_t N>
class array {
public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;
  const_iterator begin() const { return nullptr; }
  const_iterator end() const { return nullptr; }
};

template <typename T, size_t Extent = size_t(-1)>
class span {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
};
} // namespace std

// Test cases that should trigger the check

// Case 1: Function with const reference to std::vector
void processVector(const std::vector<int>& vec) {
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: parameter 'vec' is const reference to std::vector; consider using std::span instead [modernize-use-span]
// CHECK-FIXES: void processVector(std::span<const int> vec) {
  int sum = 0;
  for (const auto& val : vec) {
    sum += val;
  }
}

// Case 2: Function with const reference to std::array
void processArray(const std::array<double, 5>& arr) {
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: parameter 'arr' is const reference to std::array; consider using std::span instead [modernize-use-span]
// CHECK-FIXES: void processArray(std::span<const double, 5> arr) {
  double sum = 0.0;
  for (const auto& val : arr) {
    sum += val;
  }
}

// Case 3: Method with const reference to std::vector
class DataProcessor {
public:
  void process(const std::vector<float>& data) {
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: parameter 'data' is const reference to std::vector; consider using std::span instead [modernize-use-span]
  // CHECK-FIXES: void process(std::span<const float> data) {
    // Process data
  }
};

// Case 4: Function with multiple parameters including const reference to std::vector
void multiParam(int a, const std::vector<char>& chars, double b) {
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: parameter 'chars' is const reference to std::vector; consider using std::span instead [modernize-use-span]
// CHECK-FIXES: void multiParam(int a, std::span<const char> chars, double b) {
  // Process chars
}

// Case 5: Non-const reference to std::vector that modifies an element
void modifyVector(std::vector<int>& vec) {
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: parameter 'vec' is reference to std::vector; consider using std::span instead [modernize-use-span]
// CHECK-FIXES: void modifyVector(std::span<int> vec) {
  vec[42] = 1;
}

// Case 5b: Non-const reference to std::array
void modifyArray(std::array<double, 5>& arr) {
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: parameter 'arr' is reference to std::array; consider using std::span instead [modernize-use-span]
// CHECK-FIXES: void modifyArray(std::span<double, 5> arr) {
  // Modify arr
}

// Test cases that should NOT trigger the check

// Case 6: Const reference to a different container
void processOtherContainer(const int& str) {
  // Process str
}

// Case 7: Value parameter
void copyVector(std::vector<int> vec) {
  // Process vec
}

// Case 8: Const value parameter
void constCopyVector(const std::vector<int> vec) {
  // Process vec
}

// Case 9: Function with rvalue reference
void moveVector(std::vector<int>&& vec) {
  // Move from vec
}

// Case 10: Template function
template <typename T>
void processTemplate(const std::vector<T>& vec) {
  // Process vec
}
