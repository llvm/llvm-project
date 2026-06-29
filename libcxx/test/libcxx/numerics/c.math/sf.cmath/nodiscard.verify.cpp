//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// Check that functions are marked [[nodiscard]]

#include <cmath>

void test() {
  // clang-format off
  // assoc_laguerre
  std::assoc_laguerre(0, 0, 0.0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::assoc_laguerref(0, 0, 0.0f); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::assoc_laguerrel(0, 0, 0.0l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}
