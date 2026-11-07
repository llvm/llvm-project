//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// Check that the iterator's functions are marked [[nodiscard]]

#include <hive>

#include "test_macros.h"

void test() {
  typedef std::hive<int> Container;
  Container c;
  c.insert(1);
  Container::iterator it        = c.begin();
  Container::const_iterator cit = c.cbegin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cit;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it.operator->();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cit.operator->();
}
