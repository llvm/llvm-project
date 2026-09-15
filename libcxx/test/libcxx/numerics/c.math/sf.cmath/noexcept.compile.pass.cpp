//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// The functions carry availability markup, so referencing them fails to compile against a
// back-deployment target whose libc++ predates them.
// XFAIL: availability-mathematical_special_functions-missing

// Check that functions are marked noexcept

#include <cmath>

void test() {
  // assoc_laguerre
  static_assert(noexcept(std::assoc_laguerre(0, 0, 0.0f)));
  static_assert(noexcept(std::assoc_laguerre(0, 0, 0.0)));
  static_assert(noexcept(std::assoc_laguerre(0, 0, 0.0l)));
  static_assert(noexcept(std::assoc_laguerre(0, 0, 0)));
  static_assert(noexcept(std::assoc_laguerref(0, 0, 0.0f)));
  static_assert(noexcept(std::assoc_laguerrel(0, 0, 0.0l)));
}
