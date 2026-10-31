//===-- Unittests for isfinite macro --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

// check if macro is defined
#ifndef isfinite
#error "isfinite macro is not defined"
#else
TEST(isfinite) {
  EXPECT_TRUE(isfinite(1.0f));
  EXPECT_TRUE(isfinite(1.0));
  EXPECT_TRUE(isfinite(1.0L));
}
#endif
