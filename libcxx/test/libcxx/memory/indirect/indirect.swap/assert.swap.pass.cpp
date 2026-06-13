//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <memory>

#include "check_assertion.h"

int main(int, char**) {
  std::indirect<int, test_allocator<int>> i1(std::allocator_arg, test_allocator<int>(1));
  std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(2));
  {
    TEST_LIBCPP_ASSERT_FAILURE(swap(i1, i2), "swapping std::indirect objects with different allocators");
  }
  {
    TEST_LIBCPP_ASSERT_FAILURE(i1.swap(i2), "swapping std::indirect objects with different allocators");
  }

  return 0;
}
