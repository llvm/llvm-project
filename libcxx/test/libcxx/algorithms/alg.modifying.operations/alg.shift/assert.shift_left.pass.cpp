//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <algorithm>
#include <array>

#include "check_assertion.h"

int main(int, char**) {
  std::array<int, 5> arr = {1, 2, 3, 4, 5};
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::shift_left(arr, -2), "n must be greater than or equal to 0");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::shift_left(arr.begin(), arr.end(), -2), "n must be greater than or equal to 0");

  return 0;
}
