//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <cmath>

// double      assoc_laguerre(unsigned n, unsigned m, double x);
// float       assoc_laguerref(unsigned n, unsigned m, float x);
// long double assoc_laguerrel(unsigned n, unsigned m, long double x);

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "common.h"
#include "type_algorithms.h"

// Each overload is fixed to a single floating-point type, so it is tested
// directly with that type. The integer-argument overload
// (assoc_laguerre(unsigned, unsigned, Integer) -> double) is not implemented yet.
template <class ExpectedFuncRet, class FuncArgX, class Func>
void test(Func assoc_laguerre) {
  // sample values
  assert(between(0.99, assoc_laguerre(0, 0, FuncArgX(0)), 1.01));
  assert(between(0.99, assoc_laguerre(1, 1, FuncArgX(1)), 1.01));
  assert(between(-1.01, assoc_laguerre(2, 0, FuncArgX(2)), -0.99));
  assert(between(-0.01, assoc_laguerre(2, 2, FuncArgX(2)), 0.01));
  if constexpr (std::is_floating_point_v<FuncArgX>)
    assert(between(60.124, assoc_laguerre(2, 10, FuncArgX(0.5)), 60.126));

  static_assert(std::is_same_v<decltype(assoc_laguerre(0, 0, FuncArgX(0))), ExpectedFuncRet>);

  // NaN argument -> NaN result, without a domain error ([sf.cmath.general]/1).
  if constexpr (std::is_floating_point_v<FuncArgX>) {
    auto test_nan = [&](FuncArgX nan) {
      check_no_domain_error([&] { assert(std::isnan(assoc_laguerre(0, 0, nan))); });
    };
    if (std::numeric_limits<FuncArgX>::has_quiet_NaN)
      test_nan(std::numeric_limits<FuncArgX>::quiet_NaN());
    if (std::numeric_limits<FuncArgX>::has_signaling_NaN)
      test_nan(std::numeric_limits<FuncArgX>::signaling_NaN());
  }

  // A negative argument is in the domain (no domain error).
  check_no_domain_error([&] { assert(between(1.99, assoc_laguerre(1, 0, FuncArgX(-1)), 2.01)); });
}

struct TestIntegral {
  template <class Int>
  void operator()() const {
    test<double, Int>([](unsigned n, unsigned m, Int x) { return std::assoc_laguerre(n, m, x); });
  }
};

int main(int, char**) {
  test<double, double>([](unsigned n, unsigned m, double x) { return std::assoc_laguerre(n, m, x); });

  test<float, float>([](unsigned n, unsigned m, float x) { return std::assoc_laguerref(n, m, x); });
  test<long double, long double>([](unsigned n, unsigned m, long double x) { return std::assoc_laguerrel(n, m, x); });

  // types::for_each(types::integral_types{}, TestIntegral{}); WIP

  return 0;
}
