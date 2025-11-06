//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <memory>

#include "check_assertion.h"

int main(int, char**) {
  std::indirect<int> i;
  auto(std::move(i));
  {
    TEST_LIBCPP_ASSERT_FAILURE(*i, "operator* called on a valueless std::indirect object");
  }
  {
    TEST_LIBCPP_ASSERT_FAILURE(*std::move(i), "operator* called on a valueless std::indirect object");
  }
  {
    TEST_LIBCPP_ASSERT_FAILURE(*std::as_const(i), "operator* called on a valueless std::indirect object");
  }
  {
    TEST_LIBCPP_ASSERT_FAILURE(*std::move(std::as_const(i)), "operator* called on a valueless std::indirect object");
  }

  return 0;
}
