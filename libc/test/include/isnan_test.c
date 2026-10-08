//===-- Unittests for isnan macro -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"
#include "test/UnitTest/LibcCTest.h"

// check if macro is defined
#ifndef isnan
#error "isnan macro is not defined"
#else
TEST(isnan) {
  EXPECT_FALSE(isnan(1.0f));
  EXPECT_FALSE(isnan(1.0));
  EXPECT_FALSE(isnan(1.0L));
}
#endif
