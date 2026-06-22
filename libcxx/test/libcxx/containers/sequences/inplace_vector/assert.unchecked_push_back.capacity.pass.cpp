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

#include <cassert>
#include <inplace_vector>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  std::inplace_vector<int, 2> c{1, 2};
  int value = 3;
  TEST_LIBCPP_ASSERT_FAILURE(
      c.unchecked_emplace_back(3),
      "inplace_vector<T,N>::unchecked_emplace_back(Args...): Adding element when size() >= capacity()");
  TEST_LIBCPP_ASSERT_FAILURE(
      c.unchecked_push_back(value),
      "inplace_vector<T,N>::unchecked_emplace_back(Args...): Adding element when size() >= capacity()");
  TEST_LIBCPP_ASSERT_FAILURE(
      c.unchecked_push_back(std::move(value)),
      "inplace_vector<T,N>::unchecked_emplace_back(Args...): Adding element when size() >= capacity()");

  return 0;
}
