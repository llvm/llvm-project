//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <cmath>

// float       assoc_laguerref(unsigned n, unsigned m, float x);
// long double assoc_laguerrel(unsigned n, unsigned m, long double x);

#include <cassert>
#include <cmath>
#include <limits>

#include "common.h"

// Each overload is fixed to a single floating-point type, so it is tested
// directly with that type. The integer-argument overload
// (assoc_laguerre(unsigned, unsigned, Integer) -> double) is not implemented yet.
template <class T, class Func>
void test(Func assoc_laguerre) {
  // sample values
  assert(between(0.99, assoc_laguerre(0, 0, T(0)), 1.01));
  assert(between(0.99, assoc_laguerre(1, 1, T(1)), 1.01));
  assert(between(-1.01, assoc_laguerre(2, 0, T(2)), -0.99));
  assert(between(-0.01, assoc_laguerre(2, 2, T(2)), 0.01));
  assert(between(60.124, assoc_laguerre(2, 10, T(0.5)), 60.126));

  static_assert(std::is_same_v<decltype(assoc_laguerre(0, 0, T(0))), T>);

  // NaN argument -> NaN result, without a domain error ([sf.cmath.general]/1).
  auto test_nan = [&](T nan) { check_no_domain_error([&] { assert(std::isnan(assoc_laguerre(0, 0, nan))); }); };
  if (std::numeric_limits<T>::has_quiet_NaN)
    test_nan(std::numeric_limits<T>::quiet_NaN());
  if (std::numeric_limits<T>::has_signaling_NaN)
    test_nan(std::numeric_limits<T>::signaling_NaN());

  // A negative argument is in the domain (no domain error).
  check_no_domain_error([&] { assert(between(1.99, assoc_laguerre(1, 0, T(-1)), 2.01)); });
}

int main(int, char**) {
  test<float>([](unsigned __n, unsigned __m, float __x) { return std::assoc_laguerref(__n, __m, __x); });
  test<long double>([](unsigned __n, unsigned __m, long double __x) { return std::assoc_laguerrel(__n, __m, __x); });

  return 0;
}