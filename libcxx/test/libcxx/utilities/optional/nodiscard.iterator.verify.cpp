//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// Check that functions are marked [[nodiscard]]

#include <optional>
#include <utility>

#include "test_macros.h"

void test() {
  using Container = std::optional<int>;

  Container c;
  Container::iterator it        = c.begin();
  Container::const_iterator cit = std::as_const(c).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cit;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cit[0];

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cit + 1;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + cit;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cit - 1;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cit - cit;
}
