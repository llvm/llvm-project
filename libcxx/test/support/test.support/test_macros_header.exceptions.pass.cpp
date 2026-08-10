//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure the TEST_HAS_NO_EXCEPTIONS macro is NOT defined when exceptions
// are enabled.

// UNSUPPORTED: no-exceptions

#include "test_macros.h"

#ifdef TEST_HAS_NO_EXCEPTIONS
#  error "TEST_HAS_NO_EXCEPTIONS should NOT be defined"
#endif

int main(int, char**) {
    try { (void)0; } catch (...) { }
    return 0;
}
