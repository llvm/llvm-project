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

struct NonCommonView : std::ranges::view_base {
  int* begin() const;
  const int* end() const;

  int* base();

  int* begin();
  const int* end();
};
static_assert(!std::ranges::common_range<NonCommonView>);

void test() {
  NonCommonView range;
  auto pred = [](int) { return true; };

  auto v = std::views::filter(range, pred);

  // [range.filter.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(v).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.pred();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();

  // [range.filter.iterator]

  auto it = v.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(it);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;

  // [range.filter.sentinel]

  auto st = v.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.base();

  // [range.filter.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::filter(range, pred);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::filter(pred);
}
