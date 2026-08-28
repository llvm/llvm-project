//===-- Unittests for fpclassify macro ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

// check if macro is defined
#ifndef fpclassify
#error "fpclassify macro is not defined"
#else
TEST(fpclassify) {
  EXPECT_TRUE(fpclassify(1.819f) == FP_NORMAL);
  EXPECT_TRUE(fpclassify(-1.726) == FP_NORMAL);
  EXPECT_TRUE(fpclassify(1.426L) == FP_NORMAL);
  EXPECT_TRUE(fpclassify(-0.0f) == FP_ZERO);
  EXPECT_TRUE(fpclassify(0.0) == FP_ZERO);
  EXPECT_TRUE(fpclassify(-0.0L) == FP_ZERO);
}
#endif
