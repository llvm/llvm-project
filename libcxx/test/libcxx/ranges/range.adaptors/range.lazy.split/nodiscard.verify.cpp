//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::lazy_split_view and std::views::lazy_split are marked as [[nodiscard]].

#include <ranges>
#include <string>
#include <utility>

void test() {
  std::string str = "the quick brown fox";
  char pattern    = ' ';

  auto v = std::views::lazy_split(str, pattern);

  // [range.lazy.split.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.base();
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

  // [range.lazy.split.outer]

  auto outer_it = as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *outer_it;

  // [range.lazy.split.outer.value]

  auto value = *outer_it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  value.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  value.end();

  // [range.lazy.split.outer.inner]

  auto inner_it = value.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  inner_it.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(inner_it).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *inner_it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(inner_it);

  // [range.lazy.split.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::lazy_split(str, pattern);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::lazy_split(pattern);
}
