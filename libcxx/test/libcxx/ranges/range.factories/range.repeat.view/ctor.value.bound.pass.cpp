//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-has-debug-mode && !libcpp-has-assertions
// REQUIRES: has-unix-headers
// XFAIL: availability-verbose_abort-missing

// constexpr explicit repeat_view(W&& value, Bound bound = Bound());
// constexpr explicit repeat_view(const W& value, Bound bound = Bound());

#include <ranges>

#include "check_assertion.h"

// clang-format off
int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::repeat_view(0, -1), "The value of bound must be greater than or equal to 0");
  const int val = 0;
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::repeat_view(val, -1), "The value of bound must be greater than or equal to 0");

  return 0;
}
// clang-format on
