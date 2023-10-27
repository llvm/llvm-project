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

// constexpr explicit latch(ptrdiff_t __expected);

// Make sure that calling latch with a negative value triggers an assertion

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{safe|debug}}
// XFAIL: availability-verbose_abort-missing

#include <latch>

#include "check_assertion.h"

int main(int, char **) {
  {
    TEST_LIBCPP_ASSERT_FAILURE([]{ std::latch l(-1); }(),
                               "latch::latch(ptrdiff_t): latch cannot be "
                               "initialized with a negative value");
  }

  // We can't check the precondition for max() because there's no value
  // that would violate the precondition (in our implementation)

  return 0;
}
