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

#include "../range_adaptor_types.h"
#include "test_iterators.h"

struct CommonView : std::ranges::view_interface<CommonView> {
  int* begin();
  int* begin() const;
  int* end();
  int* end() const;

  int size() const;
};
static_assert(std::ranges::common_range<CommonView>);
static_assert(std::same_as<std::ranges::iterator_t<CommonView>, std::ranges::iterator_t<const CommonView>>);
static_assert(std::same_as<std::ranges::sentinel_t<CommonView>, std::ranges::sentinel_t<const CommonView>>);

template <class... Args>
struct Invocable {
  int operator()(Args...) const { return 5; }
};

void test() {
  CommonView commonRange;
  std::ranges::zip_transform_view commonZtv{[](int x) { return x; }, commonRange};
  static_assert(std::same_as<decltype(commonZtv.end()), decltype(commonZtv.begin())>);

  // [range.zip_transform.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  commonZtv.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(commonZtv).begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  commonZtv.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(commonZtv).end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  commonZtv.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(commonZtv).size();

  // [range.zip_transform.iterator]
  {
    auto it  = commonZtv.begin();
    auto cIt = std::as_const(commonZtv).begin();

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
  }

  // [range.zip_transform.sentinel]
  {
    auto MakeTuple = [](auto&&... args) { return std::tuple(std::forward<decltype(args)>(args)...); };
    int arr[]      = {1, 2, 3};
    std::ranges::zip_transform_view nonCommonZtv{MakeTuple, std::views::iota(0, 3), ForwardSizedNonCommon(arr)};
    static_assert(!std::ranges::common_range<decltype(nonCommonZtv)>);
    static_assert(!std::same_as<decltype(nonCommonZtv.end()), decltype(nonCommonZtv.begin())>);
    auto it  = nonCommonZtv.begin();
    auto cIt = std::as_const(nonCommonZtv).begin();
    auto st  = nonCommonZtv.end();
    auto cSt = std::as_const(nonCommonZtv).end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - st;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    st - it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - cSt;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cSt - it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    cIt - st;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    st - cIt;
  }

  // [range.zip_transform.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::zip_transform(Invocable<>{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::zip_transform([](int x) { return x; }, commonRange);
}
