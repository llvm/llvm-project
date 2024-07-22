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

// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// REQUIRES: has-unix-headers

// <semaphore>

// constexpr explicit counting_semaphore(ptrdiff_t __count);

// Make sure that constructing counting_semaphore with a negative value triggers an assertion

#include <semaphore>

#include "check_assertion.h"

int main(int, char**) {
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        [] { std::counting_semaphore<> s(-1); }(),
        "counting_semaphore::counting_semaphore(ptrdiff_t): counting_semaphore cannot be "
        "initialized with a negative value");
  }
  // We can't check the precondition for max() because there's no value
  // that would violate the precondition (in our implementation)

  return 0;
}
