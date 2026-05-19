//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY

// <array>

// Check that functions are marked [[nodiscard]]

#include <array>

#include "test_macros.h"

void test() { // __static_bounded_iter
  typedef std::array<int, 94> Container;
  ASSERT_SAME_TYPE(Container::iterator, std::__static_bounded_iter<int*, 94>);

  Container::iterator it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

  std::pointer_traits<Container::iterator> pt;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  pt.to_address(it);
}
