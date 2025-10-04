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
#include <functional>
#include <vector>

#include "test_macros.h"

void test() {
  std::vector<int> range;
  std::ranges::less_equal pred;

  std::views::drop(pred);                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::split(range, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::split(1);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::take(range, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take(1);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 23
  std::views::drop(std::views::repeat(1)); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::repeat(1);                            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::repeat(1, std::unreachable_sentinel); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  auto rvalue_view = std::views::as_rvalue(range);
  std::views::as_rvalue(range);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::as_rvalue(rvalue_view); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::chunk_by(pred);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk_by(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::take(std::views::repeat(3), 3);                            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take(std::views::repeat(3, std::unreachable_sentinel), 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::allocator<int> alloc;

  std::ranges::to<std::vector<int>>(range);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector<int>>(range, alloc);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector>(range);              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector>(range, alloc);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector<int>>();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector<int>>(alloc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector>();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector>(alloc);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // TEST_STD_VER >= 23
}
