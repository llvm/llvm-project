//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::transform_view and std::views::transform are marked as [[nodiscard]].

#include <ranges>
#include <utility>
#include <functional>

#include "test_iterators.h"

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
  auto v = View{} | std::views::transform(std::identity{});

  // [range.transform.view]

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

  // [range.transform.iterator]

  auto it = v.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(it)[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 0;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  0 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 0;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  // [range.transform.sentinel]

  auto st = v.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  auto c_it = std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - c_it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c_it - st;

  // [range.transform.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(View{}, std::identity{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(std::identity{});
}
