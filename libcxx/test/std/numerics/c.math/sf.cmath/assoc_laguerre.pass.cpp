//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <cmath>
//
// [sf.cmath.assoc.laguerre], associated Laguerre polynomials
// float               assoc_laguerref(unsigned n, unsigned m, float x);

#include <cassert>
#include <cmath>
#include <limits>

#include "common.h"
#include "type_algorithms.h"

struct TestFloatingPoint {
  template <class T>
  void operator()() const {
    // sample value testing
    assert(between(0.99f, std::assoc_laguerref(0, 0, T(0.)), 1.01f));
    assert(between(0.99f, std::assoc_laguerref(1, 1, T(1.)), 1.01f));
    assert(between(-1.01f, std::assoc_laguerref(2, 0, T(2.)), -0.99f));
    assert(between(-0.01f, std::assoc_laguerref(2, 2, T(2.)), 0.01f));
    assert(between(60.124f, std::assoc_laguerref(2, 10, T(0.5)), 60.126f));

    // return type: float
    static_assert(std::is_same_v<decltype(std::assoc_laguerref(0, 0, T(0.))), float>);

    // NaN input -> NaN output (w/o domain error)
    auto check_nan = [](T nan) {
      check_no_domain_error([nan] { assert(std::isnan(std::assoc_laguerref(0, 0, nan))); });
    };
    if (std::numeric_limits<T>::has_quiet_NaN)
      check_nan(std::numeric_limits<T>::quiet_NaN());
    if (std::numeric_limits<T>::has_signaling_NaN)
      check_nan(std::numeric_limits<T>::signaling_NaN());

    // negative x: no domain error
    check_no_domain_error([] { assert(between(1.99f, std::assoc_laguerref(1, 0, T(-1)), 2.01f)); });
  }
};

struct TestIntegral {
  template <class T>
  void operator()() const {
    assert(between(0.99f, std::assoc_laguerref(0, 0, T(0.)), 1.01f));

    static_assert(std::is_same_v<decltype(std::assoc_laguerref(0, 0, T(0.))), float>);
  }
};

int main() {
  types::for_each(types::floating_point_types{}, TestFloatingPoint{});
  types::for_each(types::integral_types{}, TestIntegral{});
}
