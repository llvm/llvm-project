//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: !libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing

// <ranges>

// Call stride_view() ctor empty stride <= 0

#include "check_assertion.h"
#include <ranges>

void test() {
  int range[] = {1, 2, 3};
  // Keep up to date with assertion message from the ctor.
  TEST_LIBCPP_ASSERT_FAILURE(
      [&range] { std::ranges::stride_view sv(range, 0); }(), "The value of stride must be greater than 0");
  TEST_LIBCPP_ASSERT_FAILURE(
      [&range] { std::ranges::stride_view sv(range, -1); }(), "The value of stride must be greater than 0");
}

int main() {
  test();
  return 0;
}
