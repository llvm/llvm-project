// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-size-type %t

using size_t = decltype(sizeof(0));

namespace std {
template <typename T>
struct vector {
  size_t size() const;
  T &operator[](size_t);
  const T &operator[](size_t) const;
  void resize(size_t);
};

template <typename T>
struct basic_string {
  size_t size() const;
  size_t length() const;
  char &operator[](size_t);
};
using string = basic_string<char>;
} // namespace std

// Positive: int from .size(), used in comparison and subscript
void test_size_comparison(std::vector<int> &v) {
  int n = v.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'n' is of signed type 'int' but is initialized from and used as an unsigned value; consider using 'size_t' [modernize-use-size-type]
  // CHECK-FIXES: size_t n = v.size();
  for (int i = 0; i < n; ++i) {
  }
}

// Positive: int from .size(), used in resize()
void test_size_resize(std::vector<int> &v) {
  int s = v.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 's' is of signed type 'int'
  // CHECK-FIXES: size_t s = v.size();
  v.resize(s);
}

// Positive: int from .length(), used in subscript
void test_length_subscript(std::string &s) {
  int len = s.length();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'len' is of signed type 'int'
  // CHECK-FIXES: size_t len = s.length();
  for (int i = 0; i < len; ++i) {
    char c = s[i];
  }
}

// Negative: variable used in signed context (subtraction producing
// potentially negative value)
void test_signed_arithmetic(std::vector<int> &v) {
  int n = v.size();
  int x = n - 10; // Used in signed arithmetic
}

// Negative: variable not initialized from unsigned
void test_signed_init() {
  int n = 42;
  size_t s = n;
}

// Negative: variable is already unsigned
void test_already_unsigned(std::vector<int> &v) {
  size_t n = v.size();
}

// Negative: variable used in a context expecting signed
void negative_signed_param(int x);
void test_signed_param(std::vector<int> &v) {
  int n = v.size();
  negative_signed_param(n);
}

// Negative: unused variable (no uses to check)
void test_unused(std::vector<int> &v) {
  int n = v.size();
}
