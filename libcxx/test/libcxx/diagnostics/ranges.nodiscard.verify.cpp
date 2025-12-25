//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that ranges are marked [[nodiscard]] as a conforming extension

// UNSUPPORTED: c++03, c++11, c++14, c++17

// clang-format off

#include <ranges>
#include <vector>

#include "test_macros.h"

void test() {
  std::vector<int> range;

  std::views::split(range, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::split(1);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::take(range, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take(1);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 23
  std::views::repeat(1);                            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::repeat(1, std::unreachable_sentinel); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::take(std::views::repeat(3), 3);                            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take(std::views::repeat(3, std::unreachable_sentinel), 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // TEST_STD_VER >= 23
}
