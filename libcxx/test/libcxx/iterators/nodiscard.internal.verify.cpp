//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: libcpp-hardening-mode=none

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_BOUNDED_ITERATORS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY

// check that <iterator> functions are marked [[nodiscard]]

#include <array>
#include <string_view>
#include <vector>

#include "test_macros.h"

void test() {
  { // __bounded_iter
    typedef std::basic_string_view<char> Container;
    ASSERT_SAME_TYPE(Container::const_iterator, std::__bounded_iter<const char*>);

    Container::const_iterator it;

    std::pointer_traits<Container::const_iterator> pt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    pt.to_address(it);
  }

  { // __static_bounded_iter
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

  { // __wrap_iter
    typedef std::vector<int> Container;
    ASSERT_SAME_TYPE(Container::iterator, std::__wrap_iter<int*>);

    Container::iterator it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;
  }
}
