//===-- Unittests for issubnormal macro -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

// check if macro is defined
#ifndef issubnormal
#error "issubnormal macro is not defined"
#else
TEST(issubnormal) {
  EXPECT_TRUE(issubnormal(1.819f) == 0);
  EXPECT_TRUE(issubnormal(-1.726) == 0);
  EXPECT_TRUE(issubnormal(1.426L) == 0);
  EXPECT_TRUE(issubnormal(1e-308) == 1);
  EXPECT_TRUE(issubnormal(-1e-308) == 1);
}
#endif
