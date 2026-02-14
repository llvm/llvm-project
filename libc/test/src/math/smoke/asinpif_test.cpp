//===-- Unittests for asinpif ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asinpif.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAsinpifTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAsinpifTest, SpecialNumbers) {
  // asinpif(+0) = +0
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::asinpif(0.0f));
  // asinpif(-0) = -0
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::asinpif(-0.0f));

  // asinpif(1) = 0.5
  EXPECT_FP_EQ_ALL_ROUNDING(0.5f, LIBC_NAMESPACE::asinpif(1.0f));
  // asinpif(-1) = -0.5
  EXPECT_FP_EQ_ALL_ROUNDING(-0.5f, LIBC_NAMESPACE::asinpif(-1.0f));

  // asinpif(NaN) = NaN
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpif(FPBits::quiet_nan().get_val()));

  // asinpif(inf) = NaN (domain error)
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpif(inf));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpif(neg_inf));

  // asinpif(x) = NaN for |x| > 1
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpif(2.0f));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpif(-2.0f));
}

TEST_F(LlvmLibcAsinpifTest, KnownValues) {
  // asinpif(0.5) = asin(0.5)/pi = (pi/6)/pi = 1/6
  float result = LIBC_NAMESPACE::asinpif(0.5f);
  EXPECT_FP_EQ_WITH_TOLERANCE(result, 0x1.555556p-3f, 1); // ~0.16667

  // asinpif(-0.5) = -1/6
  result = LIBC_NAMESPACE::asinpif(-0.5f);
  EXPECT_FP_EQ_WITH_TOLERANCE(result, -0x1.555556p-3f, 1);
}
