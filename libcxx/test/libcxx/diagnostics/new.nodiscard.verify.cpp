//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <array> functions are marked [[nodiscard]]

// clang-format off

#include <new>

#include "test_macros.h"

void test() {
  ::operator new(0);                                      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ::operator new(0, std::nothrow);                        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ::operator new[](0);                                    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ::operator new[](0, std::nothrow);                      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#ifndef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
  ::operator new(0, std::align_val_t{1});                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ::operator new(0, std::align_val_t{1}, std::nothrow);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ::operator new[](0, std::align_val_t{1});               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ::operator new[](0, std::align_val_t{1}, std::nothrow); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // _LIBCPP_HAS_NO_ALIGNED_ALLOCATION

#if TEST_STD_VER >= 17
  int* ptr = nullptr;
  std::launder(ptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
