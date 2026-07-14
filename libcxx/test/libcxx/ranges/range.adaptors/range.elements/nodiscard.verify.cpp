//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::elements_view and std::views::elements are marked as [[nodiscard]].

#include <functional>
#include <map>
#include <ranges>
#include <utility>

struct View : std::ranges::view_interface<View> {
  std::tuple<int, int>* begin();
  const std::tuple<int, int>* begin() const;
  volatile std::tuple<int, int>* end();
  const volatile std::tuple<int, int>* end() const;
};
static_assert(!std::ranges::common_range<View>);
static_assert(!std::same_as<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
static_assert(!std::same_as<std::ranges::sentinel_t<View>, std::ranges::sentinel_t<const View>>);

struct CommonView : std::ranges::view_interface<View> {
  std::tuple<int, int>* begin();
  const std::tuple<int, int>* begin() const;
  std::tuple<int, int>* end();
  const std::tuple<int, int>* end() const;
};
static_assert(std::ranges::common_range<CommonView>);
static_assert(!std::same_as<std::ranges::iterator_t<CommonView>, std::ranges::iterator_t<const CommonView>>);
static_assert(!std::same_as<std::ranges::sentinel_t<CommonView>, std::ranges::sentinel_t<const CommonView>>);

void test() {
  auto v        = View{} | std::views::elements<1>;
  auto common_v = CommonView{} | std::views::elements<1>;

  // [range.elements.view]

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
  common_v.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(common_v).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).size();

  // [range.elements.iterator]

  auto it   = v.begin();
  auto c_it = std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c_it.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();

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

  // [range.elements.sentinel]

  auto st = v.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - c_it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c_it - st;

  // [range.elements.overview]

  auto historical_figures = std::map{
      std::pair{"Lovelace", 1815}, std::pair{"Turing", 1912}, std::pair{"Babbage", 1791}, std::pair{"Hamilton", 1936}};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::elements<0>(historical_figures);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::elements<1>(historical_figures);
}
