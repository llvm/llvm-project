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

struct View : std::ranges::view_interface<View> {
  int* begin();
  int* end();
  const int* begin() const;
  const int* end() const;
};

static_assert(std::ranges::sized_range<View>);
static_assert(std::ranges::sized_range<const View>);
static_assert(std::contiguous_iterator<std::ranges::iterator_t<View>>);
static_assert(std::contiguous_iterator<std::ranges::iterator_t<const View>>);
static_assert(std::ranges::forward_range<View>);
static_assert(std::ranges::forward_range<const View>);
static_assert(std::sized_sentinel_for<std::ranges::sentinel_t<View>, std::ranges::iterator_t<View>>);
static_assert(std::sized_sentinel_for<std::ranges::sentinel_t<const View>, std::ranges::iterator_t<const View>>);
static_assert(std::ranges::bidirectional_range<View>);
static_assert(std::ranges::bidirectional_range<const View>);
static_assert(std::ranges::common_range<View>);
static_assert(std::ranges::common_range<const View>);

void test() {
  View v;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.empty();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).empty();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.data();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).data();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.front();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).front();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.back();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).back();

  using Diff = std::ranges::range_difference_t<View>;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v[Diff{0}];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v)[Diff{0}];
}
