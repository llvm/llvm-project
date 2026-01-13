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

#include "test_range.h"

struct NonSimpleView {
  std::tuple<std::ptrdiff_t, int> begin();
  std::tuple<std::ptrdiff_t, int> end();
  const std::tuple<std::ptrdiff_t, int> begin() const;
  const std::tuple<std::ptrdiff_t, int> end() const;
};
static_assert(!simple_view<NonSimpleView>);
// concept __range_with_movable_references
static_assert(std::ranges::input_range<NonSimpleView>);
static_assert(std::move_constructible<std::ranges::range_reference_t<NonSimpleView>>);
static_assert(std::move_constructible<std::ranges::range_rvalue_reference_t<NonSimpleView>>);

void test() {
  // std::vector<int> range;
  // auto subrange   = std::ranges::subrange(range.begin(), range.end() - 1);
  // using SubrangeT = decltype(subrange);
  // static_assert(!simple_view<SubrangeT>);
  // // concept __range_with_movable_references
  // static_assert(std::ranges::input_range<SubrangeT>);
  // static_assert(std::move_constructible<std::ranges::range_reference_t<SubrangeT>>);
  // static_assert(std::move_constructible<std::ranges::range_rvalue_reference_t<SubrangeT>>);
  NonSimpleView range;
  std::ranges::enumerate_view ev{range};

  // [range.enumerate.view]

  // auto ev = subrange | std::views::enumerate;

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

  auto it = ev.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it).index();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(it);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it)[2];

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(it);

  // [range.enumerate.sentinel]

  auto st = ev.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(st).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  // [range.enumerate.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::enumerate(range);
}