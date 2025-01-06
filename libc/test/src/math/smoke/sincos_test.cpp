//===-- Unittests for sincos ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sincos.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSincosTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcSincosTest, SpecialNumbers) {
  double sin_x, cos_x;

  LIBC_NAMESPACE::sincos(aNaN, &sin_x, &cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, sin_x);

  LIBC_NAMESPACE::sincos(zero, &sin_x, &cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(0.0, sin_x);

  LIBC_NAMESPACE::sincos(neg_zero, &sin_x, &cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, sin_x);

  LIBC_NAMESPACE::sincos(inf, &sin_x, &cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, sin_x);

  LIBC_NAMESPACE::sincos(neg_inf, &sin_x, &cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, cos_x);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, sin_x);

  LIBC_NAMESPACE::sincos(0x1.0p-28, &sin_x, &cos_x);
  EXPECT_FP_EQ(1.0, cos_x);
  EXPECT_FP_EQ(0x1.0p-28, sin_x);
}
