//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <latch>

// class latch;

// void count_down(ptrdiff_t __update = 1);

// Make sure that calling count_down with a negative value or a value
// higher than the internal counter triggers an assertion.

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{safe|debug}}
// XFAIL: availability-verbose_abort-missing

#include <latch>

#include "check_assertion.h"

int main(int, char **) {
  {
    std::latch l(5);

    TEST_LIBCPP_ASSERT_FAILURE(
        l.count_down(-10), "latch::count_down called with a negative value");
  }

  {
    std::latch l(5);

    TEST_LIBCPP_ASSERT_FAILURE(l.count_down(10),
                               "latch::count_down called with a value greater "
                               "than the internal counter");
  }

  return 0;
}
