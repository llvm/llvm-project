//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test that functions are marked as [[nodiscard]].

#include <ranges>
#include <vector>
#include <utility>

void test() {
  std::vector<int> range;

  { // [range.all.general]
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::all(range);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    range | std::views::all;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::all | std::views::all;
  }

  { // [range.owning.view]
    std::ranges::owning_view v{std::ranges::subrange(range)};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(v).base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(std::as_const(v)).base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.end();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.empty();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).empty();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.size();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).size();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.data();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(v).data();
  }

  { // [range.ref.view]
    std::ranges::ref_view v{range};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.size();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    v.data();
  }
}
