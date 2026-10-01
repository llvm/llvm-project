// RUN: %check_clang_tidy -std=c++20-or-later -check-suffixes=,MISSING %s modernize-use-std-interpolation %t \
// RUN:   -- -format-style="{BasedOnStyle: LLVM, IncludeBlocks: Regroup}"
// RUN: %check_clang_tidy -std=c++20-or-later -check-suffixes=,PRESENT %s modernize-use-std-interpolation %t-present \
// RUN:   -- -- -DINCLUDES_PRESENT
// RUN: clang-tidy %s -checks=-*,modernize-use-std-interpolation -allow-no-checks -- -std=c++11 2>&1 | count 0
// RUN: clang-tidy %s -checks=-*,modernize-use-std-interpolation -allow-no-checks -- -std=c++14 2>&1 | count 0
// RUN: clang-tidy %s -checks=-*,modernize-use-std-interpolation -allow-no-checks -- -std=c++17 2>&1 | count 0
// RUN: clang-tidy %s -checks=-*,modernize-use-std-interpolation -allow-no-checks -- -x c -std=c17 2>&1 | count 0

// CHECK-FIXES-MISSING: #include <cmath>
// CHECK-FIXES-MISSING-NEXT: #include <numeric>

#if __cplusplus >= 202002L

#ifdef INCLUDES_PRESENT
#include <cmath>
#include <numeric>
#endif
// CHECK-FIXES-PRESENT-NOT: #include
// CHECK-FIXES-PRESENT: #ifdef INCLUDES_PRESENT
// CHECK-FIXES-PRESENT-NEXT: #include <cmath>
// CHECK-FIXES-PRESENT-NEXT: #include <numeric>
// CHECK-FIXES-PRESENT-NEXT: #endif
// CHECK-FIXES-PRESENT-NOT: #include

void int_calculations(int a, int b) {
  auto sum = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum = std::midpoint(a, b);

  auto reversed_sum = (b + a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto reversed_sum = std::midpoint(b, a);

  auto difference = a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto difference = std::midpoint(a, b);

  auto reversed_addition = (b - a) / 2 + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto reversed_addition = std::midpoint(a, b);

  auto parenthesized = ((a) + ((b))) / (2);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto parenthesized = std::midpoint(a, b);

  auto parenthesized_difference = (a) + (((b) - (a)) / (2));
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto parenthesized_difference = std::midpoint(a, b);

  auto signed_boundaries = (-2147483647 - 1 + 2147483647) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto signed_boundaries = std::midpoint((-2147483647 - 1), 2147483647);

  auto odd_sum = (2 + 1) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto odd_sum = std::midpoint(2, 1);
}

void unsigned_calculations(unsigned a, unsigned b) {
  auto difference = a + (b - a) / 2U;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto difference = std::midpoint(a, b);

  auto sum_unsigned = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum_unsigned = std::midpoint(a, b);

  auto boundaries = (0U + 4294967295U) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto boundaries = std::midpoint(0U, 4294967295U);
}

void long_calculations(long a, long b) {
  auto sum_long = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum_long = std::midpoint(a, b);
}

void unsigned_long_calculations(unsigned long a, unsigned long b) {
  auto sum_unsigned_long = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum_unsigned_long = std::midpoint(a, b);
}

void long_long_calculations(long long a, long long b) {
  auto sum_long_long = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum_long_long = std::midpoint(a, b);
}

void unsigned_long_long_calculations(unsigned long long a, unsigned long long b) {
  auto sum_unsigned_long_long = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum_unsigned_long_long = std::midpoint(a, b);
}

void float_calculations(float a, float b, float t) {
  auto sum = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum = std::midpoint(a, b);

  auto floating_divisor = (a + b) / 2.0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto floating_divisor = std::midpoint(a, b);

  auto half = (a + b) * 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half = std::midpoint(a, b);

  auto half_first = 0.5f * (a + b);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_first = std::midpoint(a, b);

  auto difference = a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto difference = std::midpoint(a, b);

  auto half_difference = a + (b - a) * 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_difference = std::midpoint(a, b);

  auto half_difference_swapped = 0.5f * (b - a) + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_difference_swapped = std::midpoint(a, b);

  auto lerp_difference = a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_difference = std::lerp(a, b, t);

  auto lerp_product_swapped = a + t * (b - a);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_product_swapped = std::lerp(a, b, t);

  auto lerp_addition_swapped = (b - a) * t + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_addition_swapped = std::lerp(a, b, t);

  auto lerp_both_swapped = t * (b - a) + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_both_swapped = std::lerp(a, b, t);

  auto lerp_weighted = (1 - t) * a + t * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted = std::lerp(a, b, t);

  auto lerp_weighted_swapped = b * t + a * (1 - t);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_swapped = std::lerp(a, b, t);

  auto lerp_weighted_products = a * (1 - t) + b * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_products = std::lerp(a, b, t);

  auto lerp_weighted_reverse_sum = t * b + (1 - t) * a;
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_reverse_sum = std::lerp(a, b, t);
}

void double_calculations(double a, double b, double t) {
  auto sum = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum = std::midpoint(a, b);

  auto floating_divisor = (a + b) / 2.0;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto floating_divisor = std::midpoint(a, b);

  auto half = (a + b) * 0.5;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half = std::midpoint(a, b);

  auto half_first = 0.5 * (a + b);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_first = std::midpoint(a, b);

  auto difference = a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto difference = std::midpoint(a, b);

  auto half_difference = a + (b - a) * 0.5;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_difference = std::midpoint(a, b);

  auto half_difference_swapped = 0.5 * (b - a) + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_difference_swapped = std::midpoint(a, b);

  auto lerp_difference = a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_difference = std::lerp(a, b, t);

  auto lerp_product_swapped = a + t * (b - a);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_product_swapped = std::lerp(a, b, t);

  auto lerp_addition_swapped = (b - a) * t + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_addition_swapped = std::lerp(a, b, t);

  auto lerp_both_swapped = t * (b - a) + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_both_swapped = std::lerp(a, b, t);

  auto lerp_weighted = (1 - t) * a + t * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted = std::lerp(a, b, t);

  auto lerp_weighted_swapped = b * t + a * (1 - t);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_swapped = std::lerp(a, b, t);

  auto lerp_weighted_products = a * (1 - t) + b * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_products = std::lerp(a, b, t);

  auto lerp_weighted_reverse_sum = t * b + (1 - t) * a;
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_reverse_sum = std::lerp(a, b, t);

  auto signed_zero = (-0.0 + 0.0) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto signed_zero = std::midpoint((-0.0), 0.0);

  auto subnormal_values = (1e-320 + 2e-320) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto subnormal_values = std::midpoint(1e-320, 2e-320);
}

void long_double_calculations(long double a, long double b, long double t) {
  auto sum = (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto sum = std::midpoint(a, b);

  auto floating_divisor = (a + b) / 2.0L;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto floating_divisor = std::midpoint(a, b);

  auto half = (a + b) * 0.5L;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half = std::midpoint(a, b);

  auto half_first = 0.5L * (a + b);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_first = std::midpoint(a, b);

  auto difference = a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto difference = std::midpoint(a, b);

  auto half_difference = a + (b - a) * 0.5L;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_difference = std::midpoint(a, b);

  auto half_difference_swapped = 0.5L * (b - a) + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: auto half_difference_swapped = std::midpoint(a, b);

  auto lerp_difference = a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_difference = std::lerp(a, b, t);

  auto lerp_product_swapped = a + t * (b - a);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_product_swapped = std::lerp(a, b, t);

  auto lerp_addition_swapped = (b - a) * t + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_addition_swapped = std::lerp(a, b, t);

  auto lerp_both_swapped = t * (b - a) + a;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_both_swapped = std::lerp(a, b, t);

  auto lerp_weighted = (1 - t) * a + t * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted = std::lerp(a, b, t);

  auto lerp_weighted_swapped = b * t + a * (1 - t);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_swapped = std::lerp(a, b, t);

  auto lerp_weighted_products = a * (1 - t) + b * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_products = std::lerp(a, b, t);

  auto lerp_weighted_reverse_sum = t * b + (1 - t) * a;
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: auto lerp_weighted_reverse_sum = std::lerp(a, b, t);
}
using Real = double;
typedef int Integer;

auto aliases(const Real &a, Real &b, const Real &t) {
  return a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: return std::lerp(a, b, t);
}

auto typedef_midpoint(const Integer &a, Integer &&b) {
  return a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(a, b);
}

auto explicit_casts(int a, int b) {
  return (static_cast<double>(a) + static_cast<double>(b)) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(static_cast<double>(a), static_cast<double>(b));
}

constexpr int constexpr_midpoint(int a, int b) {
  return a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(a, b);
}

struct Base { double value; };
struct Derived : Base {};

auto inherited_members(const Base &a, const Derived &b, double t) {
  return a.value + (b.value - a.value) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: return std::lerp(a.value, b.value, t);
}

auto array_elements(double *values, double t) {
  return values[0] + (values[1] - values[0]) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: return std::lerp(values[0], values[1], t);
}

namespace geometry {
using Scalar = double;

auto namespaced(Scalar a, Scalar b, Scalar t) {
  return a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: return std::lerp(a, b, t);
}

} // namespace geometry

template <class Tag>

auto nondependent_template(int a, int b) {
  return (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(a, b);
}

template auto nondependent_template<int>(int, int);
template auto nondependent_template<double>(int, int);

template <class T> auto specialized(T a, T b) { return a + (b - a) / 2; }
template <>
auto specialized<double>(double a, double b) {
  return a + (b - a) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(a, b);
}

auto captured_lambda(int a, int b) {
  return [=] {
    return (a + b) / 2;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::midpoint' instead of manual midpoint calculation
    // CHECK-FIXES: return std::midpoint(a, b);
  };
}

auto lambda_parameters() {
  return [](double a, double b, double t) {
    return a + (b - a) * t;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::lerp' instead of manual linear interpolation
    // CHECK-FIXES: return std::lerp(a, b, t);
  };
}

auto nested_calculations(int a, int b, int c) {
  return ((a + b) / 2 + c) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return (std::midpoint(a, b) + c) / 2;
}

double read_value();
struct Convertible { operator double() const; };
struct Number {
  Number operator+(Number) const;
  Number operator-(Number) const;
  Number operator*(double) const;
  Number operator/(int) const;
};
enum Enumeration { First, Second };

void unrecognized_formulas(int i, int j, double a, double b, double t) {
  (void)((i + j) / 3);
  (void)((i - j) / 2);
  (void)(i + (j - i) / 3);
  (void)((1 - t) * a + (t + 1) * b);
  (void)((2 - t) * a + t * b);
  (void)(a + (b - t) * t);
  (void)((a + b) * 0.25);
}

void mixed_midpoint_types(int i, int j, unsigned u, double a, float f, float g) {
  (void)(i + (j - u) / 2);
  (void)((i + u) / 2);
  (void)((f + g) / 2.0);
  (void)((i + j) / 2.0);
  (void)((a + i) / 2);
}

void promoted_or_unsupported_types(short a, short b, char character,
                                   bool flag, Enumeration value) {
  (void)((a + b) / 2);
  (void)((character + character) / 2);
  (void)((flag + flag) / 2);
  (void)((value + value) / 2);
}

void mixed_interpolation_types(int i, int j, double a, double b, double t,
                               float f, float g) {
  (void)(i + (j - i) * t);
  (void)(a + (b - a) * i);
  (void)(f + (g - f) * t);
  (void)(a + (b - a) * 0.25f);
}

void overloaded_arithmetic(Number a, Number b, double t) {
  (void)((a + b) / 2);
  (void)(a + (b - a) * t);
}

void user_defined_conversion(Convertible a, double b) {
  (void)((a + b) / 2);
}

void side_effects(double a, double b, double t) {
  (void)((read_value() + b) / 2);
  (void)(read_value() + (b - read_value()) * t);
  (void)((1 - read_value()) * a + read_value() * b);
  (void)((a++ + b) / 2);
  (void)((++a + b) / 2);
  (void)(a + (b - a) * (t = 0.5));
}

void volatile_reads(volatile double &a, double b, double t) {
  (void)((a + b) / 2);
  (void)(a + (b - a) * t);
  (void)(b + (t - b) * a);
}

void unevaluated_contexts(double a, double b, double t) {
  (void)sizeof((a + b) / 2);
  (void)noexcept(a + (b - a) * t);
  using Result = decltype((a + b) / 2);
  (void)requires { (a + b) / 2; a + (b - a) * t; };
}

void named_constants(int a, int b) {
  constexpr int Divisor = 2;
  (void)((a + b) / Divisor);
}

template <class T> auto dependent(T a, T b, T t) {
  return a + (b - a) * t;
}
template auto dependent(double, double, double);
template auto dependent(float, float, float);
template <class T> auto forwarding(T &&a, T &&b) {
  return a + (b - a) / 2;
}
template <class... Ts> auto variadic(Ts... values) {
  return (((values + values) / 2) + ...);
}
template <class T> concept Interpolatable = requires(T a, T b, T t) {
  a + (b - a) * t;
};
auto generic_lambda() {
  return [](auto a, auto b) { return (a + b) / 2; };
}

#define MIDPOINT(a, b) (((a) + (b)) / 2)
#define TWO 2
#define FIRST a
#define ADD +
#define IDENTITY(value) (value)
void macro_cases(int a, int b) {
  (void)MIDPOINT(a, b);
  (void)((a + b) / TWO);
  (void)((FIRST + b) / 2);
  (void)((a ADD b) / 2);
  (void)IDENTITY((a + b) / 2);
}

namespace associated {
struct Number {};
Number operator+(Number, Number);
Number operator/(Number, int);
void adl(Number a, Number b) { (void)((a + b) / 2); }
} // namespace associated

int intentionally_truncated(int a, int b) {
  return (a + b) / 2; // NOLINT(modernize-use-std-interpolation)
}

auto comma_operand(int a, int b, int c) {
  return ((static_cast<void>(a), b) + c) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint((static_cast<void>(a), b), c);
}

auto conditional_operand(bool choose, int a, int b, int c) {
  return ((choose ? a : b) + c) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint((choose ? a : b), c);
}

auto unmatched_outer(int i, int j) {
  return i + (j + i) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return i + std::midpoint(j, i);
}

template auto forwarding<int &>(int &, int &);
template auto variadic(int, int);

auto instantiate_generic_lambda() {
  return generic_lambda()(1, 2);
}

// These operations are side-effect-free, so their exclusions must work
// independently of the side-effect guard.
struct PureConvertible {
  [[gnu::const]] operator double() const;
};
struct PureArithmetic {
  [[gnu::const]] double operator+(double) const;
};
void pure_user_defined_operations(PureConvertible converted,
                                  PureArithmetic overloaded,
                                  double a, double b) {
  (void)((converted + b) / 2);
  (void)((static_cast<double>(converted) + b) / 2);
  (void)((overloaded + a + b) / 2);
}

using geometry::Scalar;
auto using_declaration(Scalar a, Scalar b, Scalar t) {
  return a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: return std::lerp(a, b, t);
}

template <Interpolatable T>
auto constrained_concrete(T, int a, int b) {
  return (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(a, b);
}
template auto constrained_concrete(double, int, int);

void consume(int);
void consume(short);
void promotion_overload(short a, short b) {
  consume((a + b) / 2);
}

#endif // __cplusplus >= 202002L
