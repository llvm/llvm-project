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

// Make sure views::counted catches a negative count.

#include <ranges>

#include "check_assertion.h"
#include "test_iterators.h"

int main(int, char**) {
  int a[] = {1, 2, 3, 4};

  TEST_LIBCPP_ASSERT_FAILURE(
      std::views::counted(static_cast<int*>(a), -1), "views::counted(it, count) called with a negative count");
  TEST_LIBCPP_ASSERT_FAILURE(std::views::counted(random_access_iterator<int*>(a), -1),
                             "views::counted(it, count) called with a negative count");

  return 0;
}
