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

// void arrive_and_wait(ptrdiff_t __update = 1);

// Make sure that calling arrive_and_wait with a negative value triggers an assertion.

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <latch>

#include "check_assertion.h"

int main(int, char **) {
  {
    std::latch l(5);

    TEST_LIBCPP_ASSERT_FAILURE(
        l.arrive_and_wait(-10),
        "latch::arrive_and_wait called with a negative value");
  }

  return 0;
}
