//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<input_iterator I, class Proj = identity,
//          indirectly_unary_invocable<projected<I, Proj>> Fun>
//   constexpr ranges::for_each_n_result<I, Fun>
//     ranges::for_each_n(I first, iter_difference_t<I> n, Fun f, Proj proj = {});
//
// [alg.foreach] requires `n >= 0`; passing a negative count is a precondition violation.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <algorithm>
#include <array>

#include "check_assertion.h"

int main(int, char**) {
  std::array a = {1, 2, 3};

  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::for_each_n(a.begin(), -1, [](int) {}), "for_each_n requires a non-negative count");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::for_each_n(a.begin(), -10000000, [](int) {}), "for_each_n requires a non-negative count");

  return 0;
}
