//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that functions are marked [[nodiscard]]

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_RAW_STORAGE_ITERATOR
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>

#include "test_macros.h"

void test() {
  int i = 0;

  {
    std::addressof(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_temporary_buffer<int>(0);
#if _LIBCPP_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_sufficiently_aligned<2>(&i);
#endif
  }

  std::allocator<int> allocator;

  {
    std::allocator_traits<std::allocator<int> > allocator_traits;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator_traits.allocate(allocator, 1);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator_traits.allocate(allocator, 1, nullptr);
  }

  allocator.allocate(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 23
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  allocator.allocate_at_least(1);
#endif

#if TEST_STD_VER <= 17
  {
    const int ci = 0;

    allocator.address(i);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator.address(ci); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator.allocate(1, nullptr);
    allocator.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif // TEST_STD_VER <= 17

#if TEST_STD_VER >= 14
  {
    std::raw_storage_iterator<int*, int> it{nullptr};

    it.base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
}