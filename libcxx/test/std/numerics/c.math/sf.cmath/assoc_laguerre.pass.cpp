//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// XFAIL: availability-mathematical_special_functions-missing

// <cmath>

// double         assoc_laguerre(unsigned n, unsigned m, double x);
// float          assoc_laguerre(unsigned n, unsigned m, float x);
// long double    assoc_laguerre(unsigned n, unsigned m, long double x);
// float          assoc_laguerref(unsigned n, unsigned m, float x);
// long double    assoc_laguerrel(unsigned n, unsigned m, long double x);
// template <class Integer>
// double         assoc_laguerre(unsigned n, unsigned m, Integer x);

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include "assoc_laguerre_data.h"
#include "common.h"
#include "type_algorithms.h"

// How far a result may sit from its reference value, in units of the last place.
//
// Boost.Math promotes `float` to `double` and `double` to `long double` before running the
// recurrence. The widest type has nothing to promote to and loses digits to cancellation.
// Worst measured over assoc_laguerre_data.h, conversion of the reference values included:
// 0 ulp for `float`, 1 for `double` with a wider `long double`, 114 for `double` without
// one, 55 for x87 `long double`, 48 for binary128. The margin over those is for codegen the
// measurement cannot model, such as -ffp-contract folding a step into an fma.
//
// The tight tiers rely on the promotion defaults in
// libcxx/src/mathematical_special_functions.cpp. A double-double `long double` (PowerPC)
// takes the tight branch but has no uniform ulp, and is not known to run this test.
template <class Float>
constexpr Float tolerance_ulp() {
  if constexpr (std::is_same_v<Float, float>)
    return 4;
  else if constexpr (std::is_same_v<Float, double>)
    return std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits ? 4 : 256;
  else
    return 256;
}

// Tests a fixed-precision overload (assoc_laguerre / assoc_laguerref / assoc_laguerrel),
// whose argument and return type are both `Float`.
template <class Float, class Func>
void test_floating_point(Func assoc_laguerre) {
  // return type is the argument type
  static_assert(std::is_same_v<decltype(assoc_laguerre(0, 0, Float(0))), Float>);

  // sample values
  assert(between(0.99, assoc_laguerre(0, 0, Float(0)), 1.01));
  assert(between(0.99, assoc_laguerre(1, 1, Float(1)), 1.01));
  assert(between(-1.01, assoc_laguerre(2, 0, Float(2)), -0.99));
  assert(between(-0.01, assoc_laguerre(2, 2, Float(2)), 0.01));
  assert(between(60.124, assoc_laguerre(2, 10, Float(0.5)), 60.126));

  // [sf.cmath.assoc.laguerre] Returns: states the domain as x >= 0, so a negative
  // argument is a domain error ([sf.cmath.general]/1.1)
  check_domain_error([&] { assoc_laguerre(1, 0, Float(-1)); });

  // [sf.cmath.general]/2 leaves both infinities in the domain unless the Returns:
  // element excludes them, and x >= 0 excludes -inf but not +inf. The leading term of
  // L^m_n is (-1)^n x^n / n!, so L^m_n(+inf) is 1 for n == 0 and (-1)^n * inf otherwise.
  if constexpr (std::numeric_limits<Float>::has_infinity) {
    const Float inf = std::numeric_limits<Float>::infinity();

    check_domain_error([&] { assoc_laguerre(1, 0, -inf); });

    check_no_domain_error([&] { assert(assoc_laguerre(0, 0, inf) == Float(1)); });
    check_no_domain_error([&] { assert(assoc_laguerre(1, 0, inf) == -inf); });
    check_no_domain_error([&] { assert(assoc_laguerre(2, 0, inf) == inf); });
    check_no_domain_error([&] { assert(assoc_laguerre(3, 5, inf) == -inf); });

    // An infinity produced from finite arguments is a range error, not a domain error:
    // errno == ERANGE and the value is +-HUGE_VAL, again with the sign of the leading
    // term (C 7.12.1/4).
    const Float max = std::numeric_limits<Float>::max();

    check_range_error([&] { assert(assoc_laguerre(2, 0, max) == inf); });
    check_range_error([&] { assert(assoc_laguerre(3, 0, max) == -inf); });
    check_range_error([&] { assert(assoc_laguerre(4, 1, max) == inf); });
    check_range_error([&] { assert(assoc_laguerre(5, 1, max) == -inf); });
  }

  // NaN argument -> NaN result, without a domain error ([sf.cmath.general]/1)
  [[maybe_unused]] auto test_nan = [&](Float nan) {
    check_no_domain_error([&] { assert(std::isnan(assoc_laguerre(0, 0, nan))); });
  };
  if constexpr (std::numeric_limits<Float>::has_quiet_NaN)
    test_nan(std::numeric_limits<Float>::quiet_NaN());
  if constexpr (std::numeric_limits<Float>::has_signaling_NaN)
    test_nan(std::numeric_limits<Float>::signaling_NaN());

  // Generated reference values. Every row is a finite argument inside the domain, so the
  // NaN, infinity and negative-argument branches must all be missed and errno left alone.
  // A structured binding would not do here: C++17 forbids capturing one in a lambda.
  for (const AssocLaguerreTestCase& test_case : assoc_laguerre_test_data) {
    const Float x        = static_cast<Float>(test_case.x);
    const Float expected = static_cast<Float>(test_case.expected);
    check_no_domain_error([&] {
      assert(ulp_distance(assoc_laguerre(test_case.n, test_case.m, x), expected) <= tolerance_ulp<Float>());
    });
  }
}

// Tests the integer-argument overload: it promotes the argument to double and returns double.
struct TestInteger {
  template <class Integer>
  void operator()() const {
    static_assert(std::is_same_v<decltype(std::assoc_laguerre(0u, 0u, Integer(0))), double>);

    // same result as the double overload with the argument cast to double
    for (Integer x : {Integer(0), Integer(1), Integer(2)})
      assert(std::assoc_laguerre(2, 0, x) == std::assoc_laguerre(2, 0, static_cast<double>(x)));
  }
};

int main(int, char**) {
  test_floating_point<float>([](unsigned n, unsigned m, float x) { return std::assoc_laguerref(n, m, x); });
  test_floating_point<float>([](unsigned n, unsigned m, float x) { return std::assoc_laguerre(n, m, x); });
  test_floating_point<double>([](unsigned n, unsigned m, double x) { return std::assoc_laguerre(n, m, x); });
  test_floating_point<long double>([](unsigned n, unsigned m, long double x) { return std::assoc_laguerrel(n, m, x); });
  test_floating_point<long double>([](unsigned n, unsigned m, long double x) { return std::assoc_laguerre(n, m, x); });

  types::for_each(types::integral_types{}, TestInteger{});

  return 0;
}
