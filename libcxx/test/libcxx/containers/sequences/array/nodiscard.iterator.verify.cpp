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

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY)
  ASSERT_SAME_TYPE(Container::iterator, std::__static_bounded_iter<int*, 94>);
#elif defined(_LIBCPP_ABI_USE_WRAP_ITER_IN_STD_ARRAY)
  ASSERT_SAME_TYPE(Container::iterator, std::__wrap_iter<int*>);
#else
  ASSERT_SAME_TYPE(Container::iterator, int*);
#endif

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY) || defined(_LIBCPP_ABI_USE_WRAP_ITER_IN_STD_ARRAY)
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
#else
  // expected-warning@+1 {{expression result unused}}
  *it;

  // expected-warning@+1 {{expression result unused}}
  it[0];

  // expected-warning@+1 {{expression result unused}}
  it + 1;

  // expected-warning@+1 {{expression result unused}}
  1 + it;

  // expected-warning@+1 {{expression result unused}}
  it - 1;

  // expected-warning@+1 {{expression result unused}}
  it - it;
#endif

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pointer_traits<Container::iterator>::to_address(it);
#endif
}
