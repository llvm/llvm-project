//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// Test that functions are marked [[nodiscard]].

#include <ranges>
#include <utility>

void test() {
  char range[6] = {'x', 'x', 'y', 'y', 'z', 'z'};
  auto view     = range | std::views::chunk(2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  view.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(std::as_const(view)).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(view).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  view.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::iter_move(view.begin());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::iter_move(std::as_const(view).begin());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  view.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::iter_move(view.end());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::iter_move(std::as_const(view).end());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *view.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(view).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (view.begin() == view.end());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (std::as_const(view).begin() == view.end());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (view.begin() == std::as_const(view).end());
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  (std::as_const(view).begin() == std::as_const(view).end());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk(3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk(range, 3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::chunk(3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::reverse | std::views::chunk(3);
}
