//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20, can-test-hardening-assertions-fast

// <ranges>
//
// template<random_access_range R = D>
//   constexpr decltype(auto) operator[](range_difference_t<R> n);
// template<random_access_range R = const D>
//   constexpr decltype(auto) operator[](range_difference_t<R> n) const;

#include <cassert>
#include <ranges>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  int a[] = {1, 2, 3, 4};

  {
    auto empty = std::ranges::subrange(a, a);
    TEST_LIBCPP_ASSERT_FAILURE(empty[0], "The index passed to `view_interface::operator[]` is out of bounds.");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::as_const(empty)[0], "The index passed to `view_interface::operator[]` is out of bounds.");
  }

  {
    auto v = std::views::take(a, 2);
    assert(v[0] == 1);
    assert(std::as_const(v)[1] == 2);
    TEST_LIBCPP_ASSERT_FAILURE(v[-1], "The index passed to `view_interface::operator[]` is out of bounds.");
    TEST_LIBCPP_ASSERT_FAILURE(v[2], "The index passed to `view_interface::operator[]` is out of bounds.");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::as_const(v)[-1], "The index passed to `view_interface::operator[]` is out of bounds.");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::as_const(v)[2], "The index passed to `view_interface::operator[]` is out of bounds.");
  }

  {
    auto unsized = std::views::iota(0);
    assert(unsized[10] == 10);
  }

  return 0;
}
