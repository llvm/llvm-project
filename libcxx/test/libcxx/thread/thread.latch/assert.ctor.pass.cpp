//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <latch>
//
// constexpr explicit latch(ptrdiff_t __expected);

// Make sure that calling latch with a negative value triggers an assertion

// REQUIRES: has-unix-headers
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <latch>

#include "check_assertion.h"

void check_latch_invalid(const std::ptrdiff_t expected) {
  std::latch l(expected);
  LIBCPP_ASSERT(false);
}

int main(int, char **) {
  { check_latch_invalid(-1) }

  return 0;
}
