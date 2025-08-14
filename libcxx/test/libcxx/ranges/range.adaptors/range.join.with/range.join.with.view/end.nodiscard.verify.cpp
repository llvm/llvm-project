//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// Test the libc++ extension that std::ranges::join_with_view::end is marked as [[nodiscard]].

#include <ranges>
#include <utility>

void test() {
  int range[3][2] = {{1, 2}, {4, 5}, {7, 8}};
  int pattern[1]  = {-3};

  std::ranges::join_with_view view(range, pattern);

  // clang-format off
  view.end(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).end(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}
