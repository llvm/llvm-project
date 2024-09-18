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

void test_builtin_elementwise_abs() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_abs(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_abs(b))>::value);
}

void test_builtin_elementwise_abs_fp() {
  const float a = -2.0f;
  float b = 1.0f;
  static_assert(!is_const<decltype(__builtin_elementwise_abs(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_abs(b))>::value);
}

void test_builtin_elementwise_add_sat() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_add_sat(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_add_sat(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_add_sat(a, a))>::value);
}

void test_builtin_elementwise_sub_sat() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_sub_sat(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_sub_sat(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_sub_sat(a, a))>::value);
}

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

void test_builtin_elementwise_max_fp() {
  const float a = 2.0f;
  float b = 1.0f;
  static_assert(!is_const<decltype(__builtin_elementwise_max(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_max(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_max(a, a))>::value);
}

void test_builtin_elementwise_min_fp() {
  const float a = 2.0f;
  float b = 1.0f;
  static_assert(!is_const<decltype(__builtin_elementwise_min(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_min(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_min(a, a))>::value);
}

void test_builtin_elementwise_ceil() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_ceil(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_ceil(b))>::value);
}

void test_builtin_elementwise_acos() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_acos(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_acos(b))>::value);
}

void test_builtin_elementwise_cos() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_cos(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_cos(b))>::value);
}

void test_builtin_elementwise_cosh() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_cosh(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_cosh(b))>::value);
}

void test_builtin_elementwise_exp() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_exp(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_exp(b))>::value);
}

void test_builtin_elementwise_exp2() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_exp2(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_exp2(b))>::value);
}

void test_builtin_elementwise_asin() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_asin(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_asin(b))>::value);
}

void test_builtin_elementwise_sin() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_sin(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_sin(b))>::value);
}

void test_builtin_elementwise_sinh() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_sinh(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_sinh(b))>::value);
}

void test_builtin_elementwise_atan() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_atan(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_atan(b))>::value);
}

void test_builtin_elementwise_tan() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_tan(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_tan(b))>::value);
}

void test_builtin_elementwise_tanh() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_tanh(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_tanh(b))>::value);
}

void test_builtin_elementwise_sqrt() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_sqrt(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_sqrt(b))>::value);
}

void test_builtin_elementwise_log() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_log(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_log(b))>::value);
}

void test_builtin_elementwise_log10() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_log10(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_log10(b))>::value);
}

void test_builtin_elementwise_log2() {
  const float a = 42.0;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_log2(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_log2(b))>::value);
}

void test_builtin_elementwise_rint() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_rint(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_rint(b))>::value);
}

void test_builtin_elementwise_nearbyint() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_nearbyint(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_nearbyint(b))>::value);
}

void test_builtin_elementwise_round() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_round(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_round(b))>::value);
}

void test_builtin_elementwise_roundeven() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_roundeven(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_roundeven(b))>::value);
}

void test_builtin_elementwise_trunc() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_trunc(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_trunc(b))>::value);
}

void test_builtin_elementwise_floor() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_floor(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_floor(b))>::value);
}

void test_builtin_elementwise_canonicalize() {
  const float a = 42.5;
  float b = 42.3;
  static_assert(!is_const<decltype(__builtin_elementwise_canonicalize(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_canonicalize(b))>::value);
}

void test_builtin_elementwise_copysign() {
  const float a = 2.0f;
  float b = -4.0f;
  static_assert(!is_const<decltype(__builtin_elementwise_copysign(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_copysign(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_copysign(a, a))>::value);
}

void test_builtin_elementwise_fma() {
  const float a = 2.0f;
  float b = -4.0f;
  float c = 1.0f;
  static_assert(!is_const<decltype(__builtin_elementwise_fma(a, a, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_fma(a, b, c))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_fma(b, a, c))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_fma(c, c, c))>::value);
}

void test_builtin_elementwise_pow() {
  const double a = 2;
  double b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_pow(a, b))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_pow(b, a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_pow(a, a))>::value);
}

void test_builtin_elementwise_bitreverse() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_bitreverse(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_bitreverse(b))>::value);  
}

void test_builtin_elementwise_popcount() {
  const int a = 2;
  int b = 1;
  static_assert(!is_const<decltype(__builtin_elementwise_popcount(a))>::value);
  static_assert(!is_const<decltype(__builtin_elementwise_popcount(b))>::value);  
}

