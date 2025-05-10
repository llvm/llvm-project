//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Test hardening assertions for std::vector<bool>.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"

template <class Allocator>
void test() {
  std::vector<bool, Allocator> c;
  TEST_LIBCPP_ASSERT_FAILURE(c.front(), "vector<bool>::front() called on an empty vector");
  TEST_LIBCPP_ASSERT_FAILURE(c.back(), "vector<bool>::back() called on an empty vector");
  TEST_LIBCPP_ASSERT_FAILURE(c[0], "vector<bool>::operator[] index out of bounds");
  TEST_LIBCPP_ASSERT_FAILURE(c.pop_back(), "vector<bool>::pop_back called on an empty vector");

  // Repeat the test with a const reference to test the const overloads.
  {
    const std::vector<bool, Allocator>& cc = c;
    TEST_LIBCPP_ASSERT_FAILURE(cc.front(), "vector<bool>::front() called on an empty vector");
    TEST_LIBCPP_ASSERT_FAILURE(cc.back(), "vector<bool>::back() called on an empty vector");
    TEST_LIBCPP_ASSERT_FAILURE(cc[0], "vector<bool>::operator[] index out of bounds");
  }

  c.push_back(true);
  c.push_back(false);
  c.push_back(true);
  TEST_LIBCPP_ASSERT_FAILURE(c[3], "vector<bool>::operator[] index out of bounds");
  TEST_LIBCPP_ASSERT_FAILURE(c[100], "vector<bool>::operator[] index out of bounds");

  // Repeat the test with a const reference to test the const overloads.
  {
    const std::vector<bool, Allocator>& cc = c;
    TEST_LIBCPP_ASSERT_FAILURE(cc[3], "vector<bool>::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(cc[100], "vector<bool>::operator[] index out of bounds");
  }

  TEST_LIBCPP_ASSERT_FAILURE(
      c.erase(c.end()), "vector<bool>::erase(iterator) called with a non-dereferenceable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(
      c.erase(c.begin() + 1, c.begin()), "vector<bool>::erase(iterator, iterator) called with an invalid range");
}

int main(int, char**) {
  test<std::allocator<bool>>();
  test<min_allocator<bool>>();

  return 0;
}
