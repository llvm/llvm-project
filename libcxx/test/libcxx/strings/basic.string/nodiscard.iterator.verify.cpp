//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Check that functions are marked [[nodiscard]]

#include <string>

#include "test_macros.h"

void test() {
  typedef std::basic_string<char> Container;

  Container c;
  Container::iterator it = c.begin();

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS)
  ASSERT_SAME_TYPE(Container::iterator, std::__bounded_iter<char*>);
#else
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 1;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pointer_traits<Container::iterator>::to_address(it);
#endif
}
