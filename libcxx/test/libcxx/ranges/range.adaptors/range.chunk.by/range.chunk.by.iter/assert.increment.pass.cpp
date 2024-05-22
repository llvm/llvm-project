//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing

// <ranges>

// Increment past end chunk_by_view iterator

#include <functional>
#include <ranges>

#include "check_assertion.h"

int main(int, char**) {
  int input[] = {1, 2, 3};
  auto view   = std::views::chunk_by(input, std::less{});
  auto it     = view.begin();
  ++it;
  TEST_LIBCPP_ASSERT_FAILURE(++it, "Trying to increment past end chunk_by_view iterator.");
  return 0;
}
