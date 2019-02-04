//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// "support/test_macros.hpp"

// #define TEST_HAS_NO_RTTI

#include "test_macros.h"

struct A { virtual ~A() {} };
struct B : A {};

int main(int, char**) {
#if defined(TEST_HAS_NO_RTTI)
    A* ptr = new B;
    (void)dynamic_cast<B*>(ptr); // expected-error{{cannot use dynamic_cast}}
#else
    A* ptr = new B;
    (void)dynamic_cast<B*>(ptr);
#error RTTI enabled
// expected-error@-1{{RTTI enabled}}
#endif

  return 0;
}
