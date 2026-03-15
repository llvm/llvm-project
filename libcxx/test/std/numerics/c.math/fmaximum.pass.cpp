//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <cmath>

// constexpr float       fmaximum(float x, float y);
// constexpr double      fmaximum(double x, double y);
// constexpr long double fmaximum(long double x, long double y);

#include <cmath>
#include <cassert>
#include <limits>
#include <type_traits>

#include "test_macros.h"

template <typename T>
constexpr bool test() {
  if (!TEST_IS_CONSTANT_EVALUATED) {
    ASSERT_SAME_TYPE(T, decltype(std::fmaximum(T(), T())));
    ASSERT_NOEXCEPT(std::fmaximum(T(), T()));
  }

  constexpr T nan = std::numeric_limits<T>::quiet_NaN();
  constexpr T inf = std::numeric_limits<T>::infinity();

  // Basic maximum
  assert(std::fmaximum(T(1.0), T(2.0)) == T(2.0));
  assert(std::fmaximum(T(2.0), T(1.0)) == T(2.0));
  assert(std::fmaximum(T(-1.0), T(-2.0)) == T(-1.0));
  assert(std::fmaximum(T(-2.0), T(-1.0)) == T(-1.0));

  // Test case from atomic failure: 5.0 vs 10.0
  assert(std::fmaximum(T(5.0), T(10.0)) == T(10.0));
  assert(std::fmaximum(T(10.0), T(5.0)) == T(10.0));

  // NaN handling: propagate NaN
  assert(__builtin_isnan(std::fmaximum(nan, T(1.0))));
  assert(__builtin_isnan(std::fmaximum(T(1.0), nan)));
  assert(__builtin_isnan(std::fmaximum(nan, T(-1.0))));
  assert(__builtin_isnan(std::fmaximum(T(-1.0), nan)));

  // Both NaN: return NaN
  assert(__builtin_isnan(std::fmaximum(nan, nan)));

  // Signed zero handling: -0.0 < +0.0, so max returns +0.0
  assert(std::fmaximum(T(-0.0), T(+0.0)) == T(+0.0));
  assert(std::fmaximum(T(+0.0), T(-0.0)) == T(+0.0));
  assert(!__builtin_signbit(std::fmaximum(T(-0.0), T(+0.0))));
  assert(!__builtin_signbit(std::fmaximum(T(+0.0), T(-0.0))));

  // Infinity
  assert(std::fmaximum(inf, T(1.0)) == inf);
  assert(std::fmaximum(T(1.0), inf) == inf);
  assert(std::fmaximum(-inf, T(1.0)) == T(1.0));
  assert(std::fmaximum(T(1.0), -inf) == T(1.0));
  assert(std::fmaximum(-inf, inf) == inf);
  assert(std::fmaximum(inf, -inf) == inf);

  // NaN with infinity: propagate NaN
  assert(__builtin_isnan(std::fmaximum(nan, inf)));
  assert(__builtin_isnan(std::fmaximum(inf, nan)));
  assert(__builtin_isnan(std::fmaximum(nan, -inf)));
  assert(__builtin_isnan(std::fmaximum(-inf, nan)));

  return true;
}

int main(int, char**) {
  // TODO: Enable static_assert on macOS once __builtin_signbit is constexpr-compatible
#ifndef __APPLE__
  static_assert(test<float>());
  static_assert(test<double>());
  static_assert(test<long double>());
#endif

  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
