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
  std::inplace_vector<int, 4> c;
  TEST_LIBCPP_ASSERT_FAILURE(c.front(), "inplace_vector<T,N>::front(): access with size() == 0");
  TEST_LIBCPP_ASSERT_FAILURE(std::as_const(c).front(), "inplace_vector<T,N>::front() const: access with size() == 0");

  return 0;
}
