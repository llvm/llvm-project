//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <utility>

#include "test_range.h"

struct NonCommonView : std::ranges::view_base {
  int* begin() const;
  const int* end() const;

  int* base();

  int* begin();
  const int* end();
};
static_assert(!std::ranges::common_range<NonCommonView>);

struct NonSimpleView : std::ranges::view_base {
  int* begin() const;
  int* end() const;

  const int* begin();
  const int* end();

  constexpr std::size_t size() { return 0; };
};
static_assert(!simple_view<NonSimpleView>);

void test() {
  NonCommonView range;
  auto pred = [](int) { return true; };

  auto nsv = std::views::take_while(NonSimpleView{}, pred);
  auto v   = std::views::take_while(range, pred);

  // [range.take_while.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(v).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.pred();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  nsv.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  nsv.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).end();

  // [range.take_while.sentinel]

  auto st = v.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.base();

  // [range.take_while.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take_while(range, pred);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take_while(pred);
}
