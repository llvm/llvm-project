//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// REQUIRES: libcpp-hardening-mode={{fast|extensive|debug}}
// XFAIL:libcpp-hardening-mode=debug && availability-verbose_abort-missing

// constexpr __iterator& operator+=(difference_type __n)

#include <ranges>

#include "check_assertion.h"

int main(int, char**) {
  int range[] = {1, 2, 3};
  auto view   = std::ranges::views::stride(range, 2);
  auto it     = view.begin();
  TEST_LIBCPP_ASSERT_FAILURE(it += 3, "Advancing the iterator beyond the end is not allowed.");

  return 0;
}
