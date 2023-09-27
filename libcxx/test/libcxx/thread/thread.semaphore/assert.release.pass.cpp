//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// REQUIRES: libcpp-hardening-mode={{safe|debug}}

// XFAIL: availability-verbose_abort-missing

// REQUIRES: has-unix-headers

// <semaphore>

// void release(ptrdiff_t __update = 1);

// Make sure that calling release with a negative value triggers or with a value
// greater than expected triggers an assertion

#include <semaphore>

#include "check_assertion.h"

int main(int, char**) {
  {
    std::counting_semaphore<> s(2);
    TEST_LIBCPP_ASSERT_FAILURE(s.release(-1), "counting_semaphore:release called with a negative value");
  }

  {
    // Call release with an arbitrary larger than expected value
    std::counting_semaphore<> s(2);
    TEST_LIBCPP_ASSERT_FAILURE(
        s.release(std::counting_semaphore<>::max()), "update is greater than the expected value");
  }

  return 0;
}
