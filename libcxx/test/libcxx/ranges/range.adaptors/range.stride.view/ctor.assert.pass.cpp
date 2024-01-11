//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing

// constexpr explicit stride_view(_View, range_difference_t<_View>)

#include <ranges>

#include "check_assertion.h"

int main(int, char**) {
  int range[] = {1, 2, 3};
  TEST_LIBCPP_ASSERT_FAILURE(
      [&range] { std::ranges::stride_view sv(range, 0); }(), "The value of stride must be greater than 0");
  TEST_LIBCPP_ASSERT_FAILURE(
      [&range] { std::ranges::stride_view sv(range, -1); }(), "The value of stride must be greater than 0");

  return 0;
}
