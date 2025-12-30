//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// Check that functions are marked [[nodiscard]]

#include <new>

#include "test_macros.h"

void test() {
  {
    std::bad_alloc ex;

    ex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::bad_array_new_length ex;

    ex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0, std::nothrow);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0, std::nothrow);
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0, std::align_val_t{1});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0, std::align_val_t{1}, std::nothrow);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0, std::align_val_t{1});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0, std::align_val_t{1}, std::nothrow);
#endif // _LIBCPP_HAS_ALIGNED_ALLOCATION
  }

#if TEST_STD_VER >= 17
  {
    int* ptr = nullptr;

    std::launder(ptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
}
