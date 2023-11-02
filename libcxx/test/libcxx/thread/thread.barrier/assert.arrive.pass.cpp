//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}

// XFAIL: availability-verbose_abort-missing

// REQUIRES: has-unix-headers

// <barrier>

// class barrier;

// void arrive(ptrdiff_t __update = 1);

// Make sure that calling arrive with a negative value or with a value greater than 'expected' triggers an assertion

#include <barrier>

#include "check_assertion.h"

int main(int, char**) {
  {
    std::barrier<> b(5);
    TEST_LIBCPP_ASSERT_FAILURE(b.arrive(0), "barrier:arrive must be called with a value greater than 0");
  }

  {
    std::barrier<> b(5);
    TEST_LIBCPP_ASSERT_FAILURE(b.arrive(10), "update is greater than the expected count for the current barrier phase");
  }

  return 0;
}
