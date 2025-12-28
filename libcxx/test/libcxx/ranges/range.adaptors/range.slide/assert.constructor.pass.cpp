//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std=at-least-c++23
// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <ranges>

// Trying to construct a slide_view with slide size <= 0

#include <ranges>
#include <vector>

#include "check_assertion.h"

int main() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6};

  TEST_LIBCPP_ASSERT_FAILURE(vector | std::views::slide(0), "Trying to construct a slide_view with slide size <= 0");
  TEST_LIBCPP_ASSERT_FAILURE(vector | std::views::slide(-1), "Trying to construct a slide_view with slide size <= 0");
}
