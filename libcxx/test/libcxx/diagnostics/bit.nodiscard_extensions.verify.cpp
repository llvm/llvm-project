//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// check that <bit> functions are marked [[nodiscard]]

#include <bit>

#include "test_macros.h"

void func() {
  std::bit_cast<unsigned int>(42); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bit_ceil(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bit_floor(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bit_width(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 23
  std::byteswap(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  std::countl_zero(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::countl_one(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::countr_zero(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::countr_one(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::has_single_bit(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::popcount(0u); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
