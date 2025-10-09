//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <bitset>

// Test hardening assertions for std::bitset.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// UNSUPPORTED: c++03
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <bitset>

#include "check_assertion.h"

int main(int, char**) {
  { // Empty bitset
    std::bitset<0> c;
    const auto& const_c = c;
    TEST_LIBCPP_ASSERT_FAILURE(c[0], "bitset::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(const_c[0], "bitset::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(c[42], "bitset::operator[] index out of bounds");
    TEST_LIBCPP_ASSERT_FAILURE(const_c[42], "bitset::operator[] index out of bounds");
  }

  { // Non-empty bitset
    std::bitset<4> c(42);
    const auto& const_c = c;
    (void)c[3]; // Check that there's no assertion on valid access.
    TEST_LIBCPP_ASSERT_FAILURE(c[4], "bitset::operator[] index out of bounds");
    (void)const_c[3]; // Check that there's no assertion on valid access.
    TEST_LIBCPP_ASSERT_FAILURE(const_c[4], "bitset::operator[] index out of bounds");
  }

  return 0;
}
