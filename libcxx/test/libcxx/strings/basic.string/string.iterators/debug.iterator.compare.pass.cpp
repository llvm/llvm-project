//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Compare iterators from different containers with <.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::string S;
    S s1;
    S s2;
    TEST_LIBCPP_ASSERT_FAILURE(s1.begin() < s2.begin(), "Attempted to compare incomparable iterators");
  }

  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
    S s1;
    S s2;
    TEST_LIBCPP_ASSERT_FAILURE(s1.begin() < s2.begin(), "Attempted to compare incomparable iterators");
  }

  return 0;
}
