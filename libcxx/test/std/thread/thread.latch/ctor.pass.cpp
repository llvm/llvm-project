//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// Until we drop support for the synchronization library in C++11/14/17
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <latch>

// inline constexpr explicit latch(ptrdiff_t __expected);

// Make sure that the ctor of latch is constexpr and explicit.

#include <latch>

#include "test_convertible.h"

static_assert(!test_convertible<std::latch, std::ptrdiff_t>(), "This constructor must be explicit");

constexpr bool test() {
  std::latch l(5);
  (void)l;
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
