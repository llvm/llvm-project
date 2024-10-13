//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Make sure we assert when erase(position) is called with an iterator that isn't a
// valid iterator into the current string.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

template <class String>
void test() {
  {
    String s("123");
    TEST_LIBCPP_ASSERT_FAILURE(
        s.erase(s.end()),
        "string::erase(iterator) called with an iterator that isn't a valid iterator into this string");
  }

  {
    String s1("123");
    String s2("456");
    TEST_LIBCPP_ASSERT_FAILURE(
        s1.erase(s2.begin()),
        "string::erase(iterator) called with an iterator that isn't a valid iterator into this string");
  }
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
