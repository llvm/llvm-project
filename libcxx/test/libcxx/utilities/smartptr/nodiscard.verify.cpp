//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <memory>

// Check that functions are marked [[nodiscard]]

#include <memory>

#include "test_macros.h"

void test() {
#if TEST_STD_VER >= 23
  {
    std::unique_ptr<int> uPtr;
    // [inout.ptr]
    std::inout_ptr(uPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // [out.ptr]
    std::out_ptr(uPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
}
