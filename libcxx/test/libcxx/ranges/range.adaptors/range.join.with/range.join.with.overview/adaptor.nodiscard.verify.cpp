//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// Test the libc++ extension that std::views::join_with is marked as [[nodiscard]].

#include <ranges>

void test() {
  int range[3][3]     = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  int pattern_base[2] = {-1, -1};
  auto pattern        = std::views::all(pattern_base);

  // clang-format off
  std::views::join_with(pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join_with(range, pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::join_with(pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::reverse | std::views::join_with(pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::join_with(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join_with(range, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::join_with(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::reverse | std::views::join_with(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}
