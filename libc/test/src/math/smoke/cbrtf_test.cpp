//===-- Unittests for cbrtf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cbrtf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCbrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcCbrtfTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cbrtf(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::cbrtf(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::cbrtf(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::cbrtf(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::cbrtf(neg_zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::cbrtf(1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::cbrtf(-1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0f, LIBC_NAMESPACE::cbrtf(8.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-2.0f, LIBC_NAMESPACE::cbrtf(-8.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(3.0f, LIBC_NAMESPACE::cbrtf(27.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-3.0f, LIBC_NAMESPACE::cbrtf(-27.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(5.0f, LIBC_NAMESPACE::cbrtf(125.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-5.0f, LIBC_NAMESPACE::cbrtf(-125.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(0x1.0p42f, LIBC_NAMESPACE::cbrtf(0x1.0p126f));
  EXPECT_FP_EQ_ALL_ROUNDING(-0x1.0p42f, LIBC_NAMESPACE::cbrtf(-0x1.0p126f));
}
