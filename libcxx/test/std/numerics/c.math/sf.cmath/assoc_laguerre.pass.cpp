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
#include <cerrno>
#include <cfenv>
#include <cmath>

#include "common.h"
#include "type_algorithms.h"

struct TestFloatingPoint {
  template <class T>
  void operator()() const {
    assert(between(0.99f, std::assoc_laguerref(0, 0, T(0.)), 1.01f));

    assert(between(0.99f, std::assoc_laguerref(1, 1, T(1.)), 1.01f));

    assert(between(-0.01f, std::assoc_laguerref(2, 2, T(2.)), 0.01f));

    assert(std::abs(std::assoc_laguerref(2, 10, 0.5f) - 60.125f) < 0.001f);

    static_assert(std::is_same_v<decltype(std::assoc_laguerref(0, 0, T(0.))), float>);

    check_no_domain_error([] { (void)std::assoc_laguerref(0, 0, std::numeric_limits<T>::quiet_NaN()); });
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
