//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <numeric>

// template <class _Fp>
// _Fp midpoint(_Fp __a, _Fp __b) noexcept

#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <numeric>

#include "test_macros.h"
#include "fp_compare.h"

template <typename T>
constexpr bool is_nan(T x) {
  return x != x;
}

template <typename T>
constexpr T get_error_pct() {
  if constexpr (std::same_as<T, float>)
    return 1.0e-4f;
  else if constexpr (std::same_as<T, double>)
    return 1.0e-12;
  else
    return 1.0e-13l;
}

template <typename T>
constexpr bool check_near(T a, T b, T expect) {
  if (std::is_constant_evaluated())
    return true;
  return fptest_close_pct(std::midpoint(a, b), expect, get_error_pct<T>());
}

template <typename T>
constexpr bool check_exact(T a, T b, T expect) {
  T res = std::midpoint(a, b);
  if (is_nan(expect)) {
    return is_nan(res);
  } else {
    return res == expect;
  }
}

template <std::floating_point T>
constexpr bool test_ppc_edge_cases() {
  if (std::is_constant_evaluated())
    return true;

// For 128 bit long double implemented as 2 doubles on PowerPC,
// nextafterl() of libm gives imprecise results which fails the
// midpoint() tests below. So skip the test for this case.
#if defined(__PPC__) && (defined(__LONG_DOUBLE_128__) && __LONG_DOUBLE_128__) &&                                       \
    !(defined(__LONG_DOUBLE_IEEE128__) && __LONG_DOUBLE_IEEE128__)
  if constexpr (sizeof(T) == 16)
    return true;
#endif

  T d1 = 3.14;
  T d0 = std::nextafter(d1, T{2});
  T d2 = std::nextafter(d1, T{5});

  auto verify = [](T res, T low, T high) { return (res == low || res == high) && (low <= res && res <= high); };

  return verify(std::midpoint(d0, d1), d0, d1) && verify(std::midpoint(d1, d2), d1, d2);
}

template <typename T>
constexpr bool test_floating_points() {
  ASSERT_SAME_TYPE(T, decltype(std::midpoint(T{}, T{})));
  ASSERT_NOEXCEPT(std::midpoint(T{}, T{}));

  constexpr T max_v      = std::numeric_limits<T>::max();
  constexpr T min_v      = std::numeric_limits<T>::min();
  constexpr T denorm_min = std::numeric_limits<T>::denorm_min();
  constexpr T inf        = std::numeric_limits<T>::infinity();
  constexpr T qnan       = std::numeric_limits<T>::quiet_NaN();

  // Things that can be compared exactly
  assert(check_exact<T>(0, 0, 0));
  assert(check_exact<T>(2, 4, 3));
  assert(check_exact<T>(4, 2, 3));
  assert(check_exact<T>(3, 4, 3.5));
  assert(check_exact<T>(0, 0.4, 0.2));
  assert(check_exact<T>(-2, -4, -3));
  assert(check_exact<T>(-2, 2, 0));
  assert(check_exact<T>(2, -2, 0));

  // Infinity
  assert(check_exact<T>(inf, inf, inf));
  assert(check_exact<T>(-inf, -inf, -inf));
  if (!std::is_constant_evaluated()) {
    assert(check_exact<T>(inf, -inf, qnan));
  }
  assert(check_exact<T>(inf, 0, inf));

  // NaN
  if (!std::is_constant_evaluated()) {
    assert(check_exact<T>(qnan, 0, qnan));
    assert(check_exact<T>(qnan, qnan, qnan));
  }

  // Subnormal
  assert(check_exact<T>(denorm_min, 0, 0));
  assert(check_exact<T>(denorm_min, denorm_min, denorm_min));

  // Things that can't be compared exactly
  assert(check_near<T>(1.3, 11.4, 6.35));
  assert(check_near<T>(11.33, 31.45, 21.39));
  assert(check_near<T>(-1.3, 11.4, 5.05));
  assert(check_near<T>(11.4, -1.3, 5.05));
  assert(check_near<T>(11.2345, 14.5432, 12.88885));
  assert(check_near<T>(2.71828182845904523536028747135266249775724709369995,
                       3.14159265358979323846264338327950288419716939937510,
                       2.92993724102441923691146542731608269097720824653752));

  assert(check_near<T>(max_v, 0, max_v / 2));
  assert(check_near<T>(0, max_v, max_v / 2));
  assert(check_near<T>(min_v, 0, min_v / 2));
  assert(check_near<T>(0, min_v, min_v / 2));
  assert(check_near<T>(max_v, max_v, max_v));
  assert(check_near<T>(min_v, min_v, min_v));
  assert(check_near<T>(max_v, min_v, max_v / 2));
  assert(check_near<T>(min_v, max_v, max_v / 2));

  // Near the min and the max
  assert(check_near<T>(max_v * 0.75, max_v * 0.50, max_v * 0.625));
  assert(check_near<T>(max_v * 0.50, max_v * 0.75, max_v * 0.625));
  assert(check_near<T>(min_v * 2, min_v * 8, min_v * 5));

  // Big numbers of different signs
  assert(check_near<T>(max_v * 0.75, max_v * -0.50, max_v * 0.125));
  assert(check_near<T>(max_v * -0.75, max_v * 0.50, max_v * -0.125));

  assert(test_ppc_edge_cases<T>());

  return true;
}

constexpr bool test() {
  test_floating_points<float>();
  test_floating_points<double>();
  test_floating_points<long double>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
