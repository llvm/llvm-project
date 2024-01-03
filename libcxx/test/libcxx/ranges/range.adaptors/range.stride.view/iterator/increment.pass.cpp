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

// constexpr stride_view::<iterator>& operator++() {

#include <ranges>

#include "check_assertion.h"

int main(int, char**) {
  int range[]   = {1, 2, 3};
  auto striv    = std::ranges::views::stride(range, 3);
  auto striv_it = striv.begin();
  ++striv_it;
  TEST_LIBCPP_ASSERT_FAILURE(striv_it++, "Cannot increment an iterator already at the end.");
  return 0;
}
