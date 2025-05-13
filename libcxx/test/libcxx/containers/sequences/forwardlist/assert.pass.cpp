//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// Test hardening assertions for std::forward_list.

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <forward_list>

#include "check_assertion.h"

int main(int, char**) {
  { // Default-constructed list.
    std::forward_list<int> c;
    const auto& const_c = c;
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "forward_list::front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(const_c.front(), "forward_list::front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(c.pop_front(), "forward_list::pop_front called on an empty list");
  }

  { // Non-empty list becomes empty.
    std::forward_list<int> c;
    const auto& const_c = c;
    c.push_front(1);

    // Check that there's no assertion on valid access.
    (void)c.front();
    (void)const_c.front();

    c.pop_front();
    TEST_LIBCPP_ASSERT_FAILURE(c.pop_front(), "forward_list::pop_front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "forward_list::front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(const_c.front(), "forward_list::front called on an empty list");
  }

  return 0;
}
