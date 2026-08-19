
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::join_view and std::views::view are marked as [[nodiscard]].

#include <functional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

struct InnerView : std::ranges::view_interface<InnerView> {
  int* begin();
  const int* begin() const;
  volatile int* end();
  const volatile int* end() const;
};

struct View : std::ranges::view_interface<View> {
  InnerView* begin();
  const InnerView* begin() const;
  volatile InnerView* end();
  const volatile InnerView* end() const;
};
static_assert(!std::ranges::common_range<View>);
static_assert(!std::same_as<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
static_assert(!std::same_as<std::ranges::sentinel_t<View>, std::ranges::sentinel_t<const View>>);

void test() {
  auto v = View{} | std::views::join;

  // [range.join.view]

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

  // [range.join.iterator]

  auto c_it = std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *c_it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(c_it);

  // [range.join.overview]

  std::vector<std::string> ss{"hello", " ", "world", "!"};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join(ss);
}
