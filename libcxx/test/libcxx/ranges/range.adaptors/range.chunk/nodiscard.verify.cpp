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

#include "types.h"

void test() {
  char range[6]     = {'x', 'x', 'y', 'y', 'z', 'z'};
  auto view         = range | std::views::chunk(2);
  auto it           = view.begin();
  int input_range[] = {1, 2, 3, 4, 5, 6};
  auto input_view   = input_span(input_range, 6) | std::views::chunk(2);
  auto outer        = input_view.begin();
  auto value        = *outer;
  auto inner        = value.begin();

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
  view.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *view.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(view).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it.base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  view.end() - it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel - it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - std::default_sentinel;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  input_view.size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *outer;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel - outer;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  outer - std::default_sentinel;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  value.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  value.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  value.size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  inner.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *inner;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel - inner;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  inner - std::default_sentinel;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::iter_move(inner);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk(3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk(range, 3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::chunk(3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::reverse | std::views::chunk(3);
}
