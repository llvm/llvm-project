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

// constexpr decltype(auto) operator*() const

#include "check_assertion.h"
#include <ranges>

int main(int, char**) {
  {
    int range[] = {1, 2, 3};
    auto view   = std::ranges::views::stride(range, 3);
    auto it     = view.begin();
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(*std::as_const(it), "Cannot dereference an iterator at the end.");
  }
  {
    int range[] = {1, 2, 3};
    auto view   = std::ranges::views::stride(range, 4);
    auto it     = view.begin();
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(*std::as_const(it), "Cannot dereference an iterator at the end.");
  }
  return 0;
}
