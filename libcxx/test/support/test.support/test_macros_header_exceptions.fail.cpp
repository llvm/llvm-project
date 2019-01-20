//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// "support/test_macros.hpp"

// #define TEST_HAS_NO_EXCEPTIONS

#include "test_macros.h"

int main() {
#if defined(TEST_HAS_NO_EXCEPTIONS)
    try { ((void)0); } catch (...) {} // expected-error {{exceptions disabled}}
#else
    try { ((void)0); } catch (...) {}
#error exceptions enabled
// expected-error@-1 {{exceptions enabled}}
#endif
}
