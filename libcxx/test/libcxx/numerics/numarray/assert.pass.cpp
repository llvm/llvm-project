//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// Test hardening assertions for std::valarray.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <valarray>

#include "check_assertion.h"

int main(int, char**) {
  { // Empty valarray
    std::valarray<int> c;
    const auto& const_c = c;
    TEST_LIBCPP_ASSERT_FAILURE(c[0], "valarray::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(const_c[0], "valarray::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[42], "valarray::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(const_c[42], "valarray::operator[] index out of bounds");
  }

  { // Non-empty valarray
    std::valarray<int> c(4);
    const auto& const_c = c;
    (void)c[3]; // Check that there's no assertion on valid access.
    TEST_LIBCPP_ASSERT_FAILURE(c[4], "valarray::operator[] index out of bounds");
    (void)const_c[3]; // Check that there's no assertion on valid access.
    TEST_LIBCPP_ASSERT_FAILURE(const_c[4], "valarray::operator[] index out of bounds");
  }

  return 0;
}
