//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <utility>
#include <vector>

void test() {
  std::vector<int> range;

  // [range.enumerate.view]

  auto ev = std::ranges::subrange(range.begin(), range.end() - 1) | std::views::enumerate;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ev.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(ev).begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ev.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(ev).end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ev.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(ev).size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ev.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(ev).base();

  // [range.enumerate.iterator]

  auto it       = ev.begin();
  auto const_it = std::as_const(ev).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it).index();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(it);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[2];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(it);

  // class enumerate_view<>::__sentinel
  auto st = ev.end();
  //   auto const_st = std::as_const(ev).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  // [range.enumerate.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::enumerate(range);
}