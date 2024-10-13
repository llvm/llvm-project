//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
//
// iterator erase(const_iterator first, const_iterator last);

// Make sure we check that the iterator range is valid and within the container.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <vector>

#include "check_assertion.h"

int main(int, char**) {
  // With first iterator from another container
  {
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {6, 7, 8, 9, 10};
    TEST_LIBCPP_ASSERT_FAILURE(
        v1.erase(v2.begin(), v1.end()),
        "vector::erase(first, last) called with an iterator range that doesn't belong to this vector");
  }

  // With last iterator from another container
  {
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {6, 7, 8, 9, 10};
    TEST_LIBCPP_ASSERT_FAILURE(
        v1.erase(v1.begin(), v2.end()),
        "vector::erase(first, last) called with a last iterator that doesn't fall within the vector");
  }

  // With both iterators from another container
  {
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {6, 7, 8, 9, 10};
    TEST_LIBCPP_ASSERT_FAILURE(
        v1.erase(v2.begin(), v2.end()),
        "vector::erase(first, last) called with an iterator range that doesn't belong to this vector");
  }

  // With an invalid range
  {
    std::vector<int> v = {1, 2, 3, 4, 5};
    TEST_LIBCPP_ASSERT_FAILURE(
        v.erase(v.begin() + 2, v.begin()), "vector::erase(first, last) called with invalid range");
  }

  return 0;
}
