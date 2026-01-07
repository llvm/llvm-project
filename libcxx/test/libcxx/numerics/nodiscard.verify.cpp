//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <numeric>

// Check that functions are marked [[nodiscard]]

#include <bit>
#include <numeric>

#include "test_macros.h"

void test() {
  // [bit.rotate]
  std::rotl(0u, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rotr(0u, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // clang-format off
#if TEST_STD_VER >= 26
  // [numeric.sat]
  std::add_sat(94, 82);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sub_sat(94, 82);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mul_sat(94, 82);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::div_sat(94, 82);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::saturate_cast<signed int>(49); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // TEST_STD_VER >= 26
  // clang-format on
}
