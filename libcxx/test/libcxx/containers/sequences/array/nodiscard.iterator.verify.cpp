//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// Check that functions are marked [[nodiscard]]

#include <array>

#include "test_macros.h"

void test() {
  typedef std::array<int, 94> Container;
  Container c;
  Container::iterator it        = c.begin();
  Container::const_iterator cit = c.cbegin();

  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  *it;
  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  *cit;

  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  it[0];
  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  cit[0];

  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  it + 1;
  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  cit + 1;

  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  1 + it;
  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  1 + cit;

  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  it - 1;
  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  cit - 1;

  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  it - it;
  // expected-warning-re@+1 {{{{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}}}
  cit - cit;
}
