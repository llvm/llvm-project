//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <algorithm>
#include <array>

#include "check_assertion.h"

int main(int, char**) {
  std::initializer_list<int> init_list{};
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::minmax(init_list),
                             "initializer_list has to contain at least one element");

  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::minmax(std::array<int, 0>{}),
                             "range has to contain at least one element");

  return 0;
}
