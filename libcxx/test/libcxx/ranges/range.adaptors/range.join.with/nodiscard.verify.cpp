//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Test that functions are marked [[nodiscard]].

#include <array>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_range.h"

void test() {
  int range[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  int pattern[2]  = {-1, -1};

  std::ranges::join_with_view view(range, pattern);

  // clang-format off
  view.base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(std::as_const(view)).base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(view).base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on

  // clang-format off
  view.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on

  // clang-format off
  view.end(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).end(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}

void test_iterator() {
  char range[3][2] = {{'x', 'x'}, {'y', 'y'}, {'z', 'z'}};
  char pattern[2]  = {',', ' '};

  std::ranges::join_with_view view(range, pattern);

  // clang-format off
  *view.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(view).begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on

  // clang-format off
  (view.begin() == view.end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (std::as_const(view).begin() == view.end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (view.begin() == std::as_const(view).end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (std::as_const(view).begin() == std::as_const(view).end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on

  // clang-format off
  iter_move(view.begin()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(std::as_const(view).begin()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}

void test_sentinel() {
  std::array<test_range<cpp20_input_iterator>, 0> range;
  std::array<int, 0> pattern;

  std::ranges::join_with_view view(range, pattern);
  static_assert(!std::ranges::common_range<decltype(view)>);

  // clang-format off
  (view.begin() == view.end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (std::as_const(view).begin() == view.end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (view.begin() == std::as_const(view).end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (std::as_const(view).begin() == std::as_const(view).end()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}

void test_overview() {
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
