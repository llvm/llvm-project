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
  Container::iterator it = c.begin();

  // expected-warning-re@+1 {{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}
  *it;

  // expected-warning-re@+1 {{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}
  it[0];

  // expected-warning-re@+1 {{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}
  it + 1;

  // expected-warning-re@+1 {{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}
  1 + it;

  // expected-warning-re@+1 {{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}
  it - 1;

  // expected-warning-re@+1 {{(ignoring return value of function declared with 'nodiscard' attribute|expression result unused)}}
  it - it;

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pointer_traits<Container::iterator>::to_address(it);
#endif
}
