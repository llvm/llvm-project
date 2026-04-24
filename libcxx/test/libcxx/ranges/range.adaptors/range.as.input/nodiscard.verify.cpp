//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//  template<input_range V>
//    requires view<V>
//  class as_input_view : public view_interface<as_input_view<V>>

// Check that functions are marked [[nodiscard]]

#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

#include "test_iterators.h"

void test() {
  std::vector<int> range;

  // [range.to.input.view]

  auto v = std::views::as_input(range);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).size();

  { // [range.to.input.iterator]

    auto it = v.begin();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(it).base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(it).base();

    auto st = v.end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    st - it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - st;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter_move(it);
  }

  // [range.to.input.overview]

  {
    struct NonCommonRange {
      auto begin() { return cpp17_input_iterator<int*>{nullptr}; };
      auto end() { return sentinel_wrapper<cpp17_input_iterator<int*>>{begin()}; };
    } nonCommonRange;
    static_assert(std::ranges::input_range<NonCommonRange>);
    static_assert(!std::ranges::common_range<NonCommonRange>);
    static_assert(!std::ranges::forward_range<NonCommonRange>);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::as_input(nonCommonRange);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::as_input(range);
  }
}
