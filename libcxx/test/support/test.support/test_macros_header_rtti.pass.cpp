//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-rtti

// "support/test_macros.hpp"

// #define TEST_HAS_NO_RTTI

#include "test_macros.h"

#if defined(TEST_HAS_NO_RTTI)
#error Macro defined unexpectedly
#endif

struct A { virtual ~A() {} };
struct B : A {};

int main(int, char**) {
    A* ptr = new B;
    (void)dynamic_cast<B*>(ptr);
    delete ptr;

  return 0;
}
