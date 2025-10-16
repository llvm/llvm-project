//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// Test the precondition check in iota_view(value, bound) that `bound` is reachable from `value`.

#include <ranges>

#include "check_assertion.h"

int main(int, char**) {
  { TEST_LIBCPP_ASSERT_FAILURE(std::ranges::iota_view(5, 0), "iota_view: bound must be reachable from value"); }
  { TEST_LIBCPP_ASSERT_FAILURE(std::ranges::iota_view(10, 5), "iota_view: bound must be reachable from value"); }

  return 0;
}
