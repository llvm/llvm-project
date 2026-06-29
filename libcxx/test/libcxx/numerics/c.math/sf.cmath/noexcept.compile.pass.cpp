//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// Check that functions are marked noexcept

#include <cmath>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestIntegral {
  template <class T>
  void operator()() const {
    ASSERT_NOEXCEPT(std::assoc_laguerre(0, 0, T{}));
  }
};

void test() {
  // assoc_laguerre
  ASSERT_NOEXCEPT(std::assoc_laguerre(0, 0, 0.0));

  ASSERT_NOEXCEPT(std::assoc_laguerref(0, 0, 0.0f));
  ASSERT_NOEXCEPT(std::assoc_laguerrel(0, 0, 0.0l));

  types::for_each(types::integral_types{}, TestIntegral{});
}
