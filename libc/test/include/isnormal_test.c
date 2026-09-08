//===-- Unittests for isnormal macro --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

// check if macro is defined
#ifndef isnormal
#error "isnormal macro is not defined"
#else
TEST(isnormal) {
  EXPECT_TRUE(isnormal(1.819f) == 1);
  EXPECT_TRUE(isnormal(-1.726) == 1);
  EXPECT_TRUE(isnormal(1.426L) == 1);
  EXPECT_TRUE(isnormal(-0.0f) == 0);
  EXPECT_TRUE(isnormal(0.0) == 0);
  EXPECT_TRUE(isnormal(-0.0L) == 0);
}
#endif
