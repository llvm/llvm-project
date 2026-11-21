//===-- Unittests for iszero macro ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

// check if macro is defined
#ifndef iszero
#error "iszero macro is not defined"
#else
TEST(iszero) {
  EXPECT_TRUE(iszero(1.0f) == 0);
  EXPECT_TRUE(iszero(1.0) == 0);
  EXPECT_TRUE(iszero(1.0L) == 0);
  EXPECT_TRUE(iszero(0.0f) == 1);
  EXPECT_TRUE(iszero(0.0) == 1);
  EXPECT_TRUE(iszero(0.0L) == 1);
}
#endif
