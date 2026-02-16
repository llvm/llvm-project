//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Test that functions are marked [[nodiscard]].

#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_range.h"

void test() {
  int range[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  int pattern[2]  = {-1, -1};

  std::ranges::join_with_view view(range, pattern);

  // [range.join.with.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(view).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  view.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  view.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(view).end();

  // [range.join.with.iterator]

  auto cIt = std::as_const(view).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cIt;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(cIt);

  // [range.join.with.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join_with(range, pattern);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join_with(pattern);
}
