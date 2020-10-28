//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure the TEST_HAS_NO_RTTI macro is defined when the no-rtti Lit feature
// is defined.

// REQUIRES: no-rtti

#include "test_macros.h"

#ifndef TEST_HAS_NO_RTTI
#  error "TEST_HAS_NO_RTTI should be defined"
#endif

struct A { virtual ~A() { } };
struct B : A { };

int main(int, char**) {
    A* ptr = new B;
    (void)dynamic_cast<B*>(ptr); // expected-error{{use of dynamic_cast requires -frtti}}
    return 0;
}
