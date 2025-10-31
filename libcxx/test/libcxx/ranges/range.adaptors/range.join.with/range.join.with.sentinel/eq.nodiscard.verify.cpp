//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// Test the libc++ extension that std::ranges::join_with_view::sentinel<Const>::operator== is marked as [[nodiscard]].

#include <array>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_range.h"

void test() {
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
