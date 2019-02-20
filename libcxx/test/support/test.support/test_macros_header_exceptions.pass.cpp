//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions

// "support/test_macros.hpp"

// #define TEST_HAS_NO_EXCEPTIONS

#include "test_macros.h"

#if defined(TEST_HAS_NO_EXCEPTIONS)
#error macro defined unexpectedly
#endif

int main(int, char**) {
    try { ((void)0); } catch (...) {}

  return 0;
}
