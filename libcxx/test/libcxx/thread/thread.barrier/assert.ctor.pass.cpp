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

// <barrier>

// class barrier;

// barrier(ptrdiff_t __count, _CompletionF __completion = _CompletionF());

// Make sure that constructing barrier with a negative value triggers an assertion

#include <barrier>

#include "check_assertion.h"

int main(int, char**) {
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        [] { std::barrier<> b(-1); }(),
        "barrier::barrier(ptrdiff_t, CompletionFunction): barrier cannot be initialized with a negative value");
  }

  {
    TEST_LIBCPP_ASSERT_FAILURE(
        [] {
          auto completion = []() {};
          std::barrier<decltype(completion)> b(-1, completion);
        }(),
        "barrier::barrier(ptrdiff_t, CompletionFunction): barrier cannot be initialized with a negative value");
  }

  // We can't check the precondition for max() because there's no value
  // that would violate the precondition (in our implementation)

  return 0;
}
