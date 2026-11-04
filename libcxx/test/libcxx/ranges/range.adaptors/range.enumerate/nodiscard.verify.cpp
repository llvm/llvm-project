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

struct View : std::ranges::view_interface<View> {
  int* begin();
  const int* begin() const;
  volatile int* end();
  const volatile int* end() const;
};
static_assert(!std::ranges::common_range<View>);
static_assert(!std::same_as<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
static_assert(!std::same_as<std::ranges::sentinel_t<View>, std::ranges::sentinel_t<const View>>);
static_assert(std::ranges::sized_range<View>);
static_assert(std::ranges::forward_range<View>);

void test() {
  View range;
  std::ranges::enumerate_view ev{range};

  // [range.enumerate.view]

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
  1 + it;
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
