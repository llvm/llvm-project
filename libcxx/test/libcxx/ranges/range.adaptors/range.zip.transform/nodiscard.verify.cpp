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

struct View : std::ranges::view_interface<View> {
  int* begin();
  const int* begin() const;
  volatile int* end();
  const volatile int* end() const;

  int size() const;
};
static_assert(!std::ranges::common_range<View>);
static_assert(!std::same_as<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
static_assert(!std::same_as<std::ranges::sentinel_t<View>, std::ranges::sentinel_t<const View>>);

template <class... Args>
struct Invocable {
  int operator()(Args...) const { return 5; }
};

void test() {
  View range;
  std::ranges::zip_transform_view ztv{[](int x) { return x; }, range};

  // [range.zip_transform.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ztv.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(ztv).begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ztv.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(ztv).end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ztv.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(ztv).size();

  // [range.zip_transform.iterator]

  auto it  = ztv.begin();
  auto cIt = std::as_const(ztv).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cIt;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cIt[0];

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  // [range.zip_transform.sentinel]

  auto st = ztv.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - st;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  st - it;

  // [range.zip_transform.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::zip_transform(Invocable<>{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::zip_transform([](int x) { return x; }, range);
}