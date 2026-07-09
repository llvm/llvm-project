//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Test the libc++ extension that std::ranges::adjacent_view and std::views::adjacent are marked as [[nodiscard]].

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

void test() {
  int range[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto v      = View{} | std::views::adjacent<2>;

  // [range.adjacent.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(v).base();

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

  // [range.adjacent.iterator]

  auto it   = v.begin();
  auto c_it = std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *c_it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c_it[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 0;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  0 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 0;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  // [range.adjacent.sentinel]

  auto st = v.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - c_it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c_it - st;

  // [range.adjacent.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::adjacent<0>(range);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::adjacent<2>(range);
}
