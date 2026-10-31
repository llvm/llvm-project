//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <ranges>

// Make sure views::take catches a negative count for the overloads that return the same range type.

#include <ranges>

#include <span>
#include <string_view>

#include "check_assertion.h"

int main(int, char**) {
  int a[] = {1, 2, 3, 4};

  TEST_LIBCPP_ASSERT_FAILURE(
      std::views::take(std::span<int>(a), -1), "views::take(range, count) called with a negative count");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::views::take(std::string_view("abcd"), -1), "views::take(range, count) called with a negative count");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::views::take(std::ranges::subrange(a, a + 4), -1), "views::take(range, count) called with a negative count");

  return 0;
}
