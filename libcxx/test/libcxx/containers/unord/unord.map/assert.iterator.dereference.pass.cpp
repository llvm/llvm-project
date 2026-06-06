//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Dereference non-dereferenceable iterator.

// REQUIRES: has-unix-headers, libcpp-hardening-mode={{extensive|debug}}
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <unordered_map>
#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::unordered_map<int, std::string> C;
    C c;
    c.insert(std::make_pair(1, "one"));
    C::iterator i = c.end();
    TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable unordered container iterator");
    C::const_iterator i2 = c.cend();
    TEST_LIBCPP_ASSERT_FAILURE(
        *i2, "Attempted to dereference a non-dereferenceable unordered container const_iterator");
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
    C::iterator i = c.end();
    TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable unordered container iterator");
    C::const_iterator i2 = c.cend();
    TEST_LIBCPP_ASSERT_FAILURE(
        *i2, "Attempted to dereference a non-dereferenceable unordered container const_iterator");
  }

  return 0;
}
