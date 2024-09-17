//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <memory> functions are marked [[nodiscard]]

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// clang-format off

#include <memory>

#include "test_macros.h"

void test() {
  std::get_temporary_buffer<int>(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_allocator_traits() {
  std::allocator<int> allocator;
  std::allocator_traits<std::allocator<int>> allocator_traits;
  allocator_traits.allocate(allocator, 1);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  allocator_traits.allocate(allocator, 1, nullptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_allocator() {
  std::allocator<int> allocator;
  allocator.allocate(1);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER <= 17
  allocator.allocate(1, nullptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
#if TEST_STD_VER >= 23
  allocator.allocate_at_least(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
