//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class InputIterator, class Size, class Function>
//    constexpr InputIterator
//    for_each_n(InputIterator first, Size n, Function f);
//
// [alg.foreach] requires `n >= 0`; passing a negative count is a precondition violation.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <algorithm>

#include "check_assertion.h"

int main(int, char**) {
  int a[] = {1, 2, 3};

  TEST_LIBCPP_ASSERT_FAILURE(std::for_each_n(a, -1, [](int) {}), "for_each_n requires a non-negative count");
  TEST_LIBCPP_ASSERT_FAILURE(std::for_each_n(a, -10000000, [](int) {}), "for_each_n requires a non-negative count");

  return 0;
}
