// RUN: %clang_cc1 %s -std=c++17 -pedantic -verify -triple=x86_64-apple-darwin9

// Simple is_const implementation.
struct true_type {
  static const bool value = true;
};

struct false_type {
  static const bool value = false;
};

template <class T> struct is_const : false_type {};
template <class T> struct is_const<const T> : true_type {};

// expected-no-diagnostics

void test_builtin_elementwise_max() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_max(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_max(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_max(a, a))>::value);
}

void test_builtin_elementwise_min() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_min(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_min(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_min(a, a))>::value);
}
