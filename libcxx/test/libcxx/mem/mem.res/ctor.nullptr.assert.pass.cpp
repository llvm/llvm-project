//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory_resource>

// Test hardening assertions for std::pmr::polymorphic_allocator.

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// We're testing nullptr assertions
// ADDITIONAL_COMPILE_FLAGS: -Wno-nonnull

#include <memory_resource>

#include "check_assertion.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::pmr::polymorphic_allocator<int>(nullptr), "Attempted to pass a nullptr resource to polymorphic_alloator");

  return 0;
}
