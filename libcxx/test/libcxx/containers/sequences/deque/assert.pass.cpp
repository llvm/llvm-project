//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Test hardening assertions for std::deque.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <deque>

#include "check_assertion.h"

int main(int, char**) {
  std::deque<int> c;
  TEST_LIBCPP_ASSERT_FAILURE(c.front(), "deque::front called on an empty deque");
  TEST_LIBCPP_ASSERT_FAILURE(c.back(), "deque::back called on an empty deque");
  TEST_LIBCPP_ASSERT_FAILURE(c[0], "deque::operator[] index out of bounds");
  TEST_LIBCPP_ASSERT_FAILURE(c.pop_front(), "deque::pop_front called on an empty deque");
  TEST_LIBCPP_ASSERT_FAILURE(c.pop_back(), "deque::pop_back called on an empty deque");

  // Repeat the test with a const reference to test the const overloads.
  {
    const std::deque<int>& cc = c;
    TEST_LIBCPP_ASSERT_FAILURE(cc.front(), "deque::front called on an empty deque");
    TEST_LIBCPP_ASSERT_FAILURE(cc.back(), "deque::back called on an empty deque");
    TEST_LIBCPP_ASSERT_FAILURE(cc[0], "deque::operator[] index out of bounds");
  }

  c.push_back(1);
  c.push_back(2);
  c.push_back(3);
  TEST_LIBCPP_ASSERT_FAILURE(c[3], "deque::operator[] index out of bounds");
  TEST_LIBCPP_ASSERT_FAILURE(c[100], "deque::operator[] index out of bounds");

  // Repeat the test with a const reference to test the const overloads.
  {
    const std::deque<int>& cc = c;
    TEST_LIBCPP_ASSERT_FAILURE(cc[3], "deque::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(cc[100], "deque::operator[] index out of bounds");
  }

  TEST_LIBCPP_ASSERT_FAILURE(c.erase(c.end()), "deque::erase(iterator) called with a non-dereferenceable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(
      c.erase(c.begin() + 1, c.begin()), "deque::erase(first, last) called with an invalid range");

  return 0;
}
