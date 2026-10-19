//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <vector>
#include <utility>

#include "../range_adaptor_types.h"
#include "test_macros.h"
#include "test_iterators.h"

void test() {
  int arr[]{94, 82, 49};
  std::vector<int> range;

  std::ranges::zip_view zv{range, arr};

  // [range.zip.view]
  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    zv.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(zv).begin();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    zv.end();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(zv).end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    zv.size();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(zv).size();
  }

  // [range.zip.iterator]

  {
    auto it  = zv.begin();
    auto cIt = std::as_const(zv).begin();

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

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter_move(it);
  }

  // [range.zip.sentinel]

  {
    std::ranges::zip_view nonCommonZv{ForwardSizedNonCommon(arr)};
    static_assert(!std::ranges::common_range<decltype(nonCommonZv)>);
    static_assert(!std::same_as<decltype(nonCommonZv.end()), decltype(nonCommonZv.begin())>);
    auto it  = nonCommonZv.begin();
    auto cIt = std::as_const(nonCommonZv).begin();
    auto st  = nonCommonZv.end();
    auto cSt = std::as_const(nonCommonZv).end();

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

  // [range.zip.overview]

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::zip();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::zip(arr, arr);
  }
}
