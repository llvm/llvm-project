//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Increment iterator past end.

// REQUIRES: has-unix-headers, libcpp-hardening-mode={{extensive|debug}}
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <unordered_set>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::unordered_set<T> C;
    C c;
    c.insert(42);
    C::iterator i = c.begin();
    assert(i != c.end());
    ++i;
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container const_iterator");
    C::const_iterator i2 = c.cbegin();
    assert(i2 != c.cend());
    ++i2;
    assert(i2 == c.cend());
    TEST_LIBCPP_ASSERT_FAILURE(++i2, "Attempted to increment a non-incrementable unordered container const_iterator");
  }

  {
    typedef int T;
    typedef std::unordered_set<T, std::hash<T>, std::equal_to<T>, min_allocator<T>> C;
    C c({42});
    C::iterator i = c.begin();
    assert(i != c.end());
    ++i;
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container const_iterator");
    C::const_iterator i2 = c.cbegin();
    assert(i2 != c.cend());
    ++i2;
    assert(i2 == c.cend());
    TEST_LIBCPP_ASSERT_FAILURE(++i2, "Attempted to increment a non-incrementable unordered container const_iterator");
  }

  return 0;
}
