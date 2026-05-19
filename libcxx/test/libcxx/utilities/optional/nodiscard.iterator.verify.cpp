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

#include "test_macros.h"

void test() {
  using Container = std::optional<int>;

  Container c;
  Container::iterator it = c.begin();

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL)
  ASSERT_SAME_TYPE(Container::iterator, std::__bounded_iter<int*>);
#else
  ASSERT_SAME_TYPE(Container::iterator, std::__capacity_aware_iterator<int*, std::optional<int>, 1>);
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL)
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

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pointer_traits<Container::iterator>::to_address(it);
#endif
}
