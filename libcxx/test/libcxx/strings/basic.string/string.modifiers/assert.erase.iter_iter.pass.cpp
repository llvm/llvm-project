//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Make sure we assert when erase(first, last) is called with iterators that aren't valid.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

template <class String>
void test() {
  // With first iterator from another container
  {
    String s1("123");
    String s2("123");
    TEST_LIBCPP_ASSERT_FAILURE(
        s1.erase(s2.cbegin(), s1.cbegin() + 1),
        "string::erase(first, last) called with an iterator range that doesn't belong to this string");
  }

  // With second iterator from another container
  {
    String s1("123");
    String s2("123");
    TEST_LIBCPP_ASSERT_FAILURE(
        s1.erase(s1.cbegin(), s2.cbegin() + 1),
        "string::erase(first, last) called with a last iterator that doesn't fall within the string");
  }

  // With both iterators from another container
  {
    String s1("123");
    String s2("123");
    TEST_LIBCPP_ASSERT_FAILURE(
        s1.erase(s2.cbegin(), s2.cbegin() + 1),
        "string::erase(first, last) called with an iterator range that doesn't belong to this string");
  }

  // With an invalid range
  {
    String s1("123");
    TEST_LIBCPP_ASSERT_FAILURE(
        s1.erase(s1.cbegin() + 1, s1.cbegin()), "string::erase(first, last) called with invalid range");
  }
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
