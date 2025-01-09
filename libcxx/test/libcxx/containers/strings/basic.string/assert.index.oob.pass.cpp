//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Index string out of bounds.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <string>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  // Test the const overloads.
  {
    using C = std::basic_string<char, std::char_traits<char>, safe_allocator<char> >;
    const C c;
    TEST_LIBCPP_ASSERT_FAILURE(c[0], "string index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[1], "string index out of bounds");
  }
  {
    using C   = std::basic_string<char, std::char_traits<char>, safe_allocator<char> >;
    const C c = "abc";
    TEST_LIBCPP_ASSERT_FAILURE(c[3], "string index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[4], "string index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[100], "string index out of bounds");
  }

  // Test the nonconst overloads.
  {
    using C = std::basic_string<char, std::char_traits<char>, safe_allocator<char> >;
    C c;
    TEST_LIBCPP_ASSERT_FAILURE(c[0], "string index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[1], "string index out of bounds");
  }
  {
    using C = std::basic_string<char, std::char_traits<char>, safe_allocator<char> >;
    C c     = "abc";
    TEST_LIBCPP_ASSERT_FAILURE(c[3], "string index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[4], "string index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[100], "string index out of bounds");
  }

  return 0;
}
