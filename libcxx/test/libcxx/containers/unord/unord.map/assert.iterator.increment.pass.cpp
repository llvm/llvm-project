//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Increment iterator past end.

// REQUIRES: has-unix-headers, libcpp-hardening-mode={{extensive|debug}}
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <unordered_map>
#include <cassert>
#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::unordered_map<int, std::string> C;
    C c;
    c.insert(std::make_pair(1, "one"));
    C::iterator i = c.begin();
    ++i;
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container iterator");
    C::const_iterator i2 = c.cbegin();
    ++i2;
    assert(i2 == c.cend());
    TEST_LIBCPP_ASSERT_FAILURE(++i2, "Attempted to increment a non-incrementable unordered container const_iterator");
  }

  {
    typedef std::unordered_map<int,
                               std::string,
                               std::hash<int>,
                               std::equal_to<int>,
                               min_allocator<std::pair<const int, std::string>>>
        C;
    C c;
    c.insert(std::make_pair(1, "one"));
    C::iterator i = c.begin();
    ++i;
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container iterator");
    C::const_iterator i2 = c.cbegin();
    ++i2;
    assert(i2 == c.cend());
    TEST_LIBCPP_ASSERT_FAILURE(++i2, "Attempted to increment a non-incrementable unordered container const_iterator");
  }

  return 0;
}
