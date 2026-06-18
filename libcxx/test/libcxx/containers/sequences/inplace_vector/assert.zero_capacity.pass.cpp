//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <inplace_vector>

#include <inplace_vector>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  std::inplace_vector<int, 0> c;

  TEST_LIBCPP_ASSERT_FAILURE((void)c[0], "inplace_vector<T,0>::operator[]: access with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE((void)std::as_const(c)[0], "inplace_vector<T,0>::operator[] const: access with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE((void)c.front(), "inplace_vector<T,0>::front(): access with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE((void)std::as_const(c).front(), "inplace_vector<T,0>::front() const: access with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE((void)c.back(), "inplace_vector<T,0>::back(): access with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE((void)std::as_const(c).back(), "inplace_vector<T,0>::back() const: access with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE(c.pop_back(), "inplace_vector<T,0>::erase(): use with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE(c.erase(c.begin()), "inplace_vector<T,0>::erase(): use with N == 0");
  TEST_LIBCPP_ASSERT_FAILURE(
      c.erase(c.begin(), c.begin() + 1), "inplace_vector<T,0>::erase(const_iterator, const_iterator): use with N == 0");

  return 0;
}
