//===-- Unittests for cbrt ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cbrt.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCbrtTest = LIBC_NAMESPACE::testing::FPTest<double>;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcCbrtTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cbrt(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::cbrt(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::cbrt(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::cbrt(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::cbrt(neg_zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::cbrt(1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0, LIBC_NAMESPACE::cbrt(-1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0, LIBC_NAMESPACE::cbrt(8.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-2.0, LIBC_NAMESPACE::cbrt(-8.0));
  EXPECT_FP_EQ_ALL_ROUNDING(3.0, LIBC_NAMESPACE::cbrt(27.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-3.0, LIBC_NAMESPACE::cbrt(-27.0));
  EXPECT_FP_EQ_ALL_ROUNDING(5.0, LIBC_NAMESPACE::cbrt(125.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-5.0, LIBC_NAMESPACE::cbrt(-125.0));
  EXPECT_FP_EQ_ALL_ROUNDING(0x1.0p42, LIBC_NAMESPACE::cbrt(0x1.0p126));
  EXPECT_FP_EQ_ALL_ROUNDING(-0x1.0p42, LIBC_NAMESPACE::cbrt(-0x1.0p126));
  EXPECT_FP_EQ_ALL_ROUNDING(0x1.0p341, LIBC_NAMESPACE::cbrt(0x1.0p1023));
  EXPECT_FP_EQ_ALL_ROUNDING(-0x1.0p341, LIBC_NAMESPACE::cbrt(-0x1.0p1023));
}
