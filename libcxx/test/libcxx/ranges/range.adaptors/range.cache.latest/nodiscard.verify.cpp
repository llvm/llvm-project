//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <utility>
#include <vector>

#include "test_macros.h"

void test() {
  std::vector<int> range;
  std::ranges::cache_latest_view clv{range};

  // [range.cache.latest.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(clv).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(clv).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  clv.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  clv.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  clv.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(clv).size();

#if TEST_STD_VER >= 26

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  clv.reserve_hint();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(clv).reserve_hint();

#endif

  // [range.cache.latest.iterator]

  auto it = clv.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *(std::as_const(it));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(it);

  // [range.cache.latest.sentinel]

  auto st = clv.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(st).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  // [range.enumerate.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::cache_latest(range);
}
