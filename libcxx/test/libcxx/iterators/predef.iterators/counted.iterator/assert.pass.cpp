//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <iterator>

#include "check_assertion.h"
#include "test_iterators.h"

int main(int, char**) {
  using Iter = std::counted_iterator<int*>;
  int a[]    = {1, 2, 3};
  Iter valid_i(a, 1);

  {
    Iter i;

    TEST_LIBCPP_ASSERT_FAILURE(*i, "Iterator is equal to or past end.");
    TEST_LIBCPP_ASSERT_FAILURE(i[999], "Subscript argument must be less than size.");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::iter_move(i), "Iterator must not be past end of range.");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::iter_swap(i, valid_i), "Iterators must not be past end of range.");
    TEST_LIBCPP_ASSERT_FAILURE(std::ranges::iter_swap(valid_i, i), "Iterators must not be past end of range.");
    std::ranges::iter_swap(valid_i, valid_i); // Ok
  }

  { // Check the `const` overload of `operator*`.
    const Iter i;

    TEST_LIBCPP_ASSERT_FAILURE(*i, "Iterator is equal to or past end.");
  }

  return 0;
}
