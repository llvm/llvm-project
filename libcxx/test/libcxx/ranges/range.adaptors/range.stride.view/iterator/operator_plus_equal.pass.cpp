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

#include "check_assertion.h"
#include <ranges>

void operator_plus_equal_past_end_is_illegal() {
  int range[]   = {1, 2, 3};
  auto striv    = std::ranges::views::stride(range, 2);
  auto striv_it = striv.begin();
  TEST_LIBCPP_ASSERT_FAILURE(striv_it += 3, "Advancing the iterator beyond the end is not allowed.");
}

int main(int, char**) {
  operator_plus_equal_past_end_is_illegal();

  return 0;
}
