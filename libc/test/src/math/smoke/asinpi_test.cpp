//===-- Unittests for asinpi ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAsinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAsinpiTest, SpecialNumbers) {
  // asinpi(+0) = +0
  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::asinpi(0.0));
  // asinpi(-0) = -0
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0, LIBC_NAMESPACE::asinpi(-0.0));

  // asinpi(1) = 0.5
  EXPECT_FP_EQ_ALL_ROUNDING(0.5, LIBC_NAMESPACE::asinpi(1.0));
  // asinpi(-1) = -0.5
  EXPECT_FP_EQ_ALL_ROUNDING(-0.5, LIBC_NAMESPACE::asinpi(-1.0));

  // asinpi(NaN) = NaN
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpi(FPBits::quiet_nan().get_val()));

  // asinpi(inf) = NaN
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpi(inf));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpi(neg_inf));

  // asinpi(x) = NaN for |x| > 1
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpi(2.0));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::asinpi(-2.0));
}

TEST_F(LlvmLibcAsinpiTest, KnownValues) {
  // asinpi(0.5) = 1/6
  double result = LIBC_NAMESPACE::asinpi(0.5);
  double expected = 0x1.5555555555555p-3; // 1/6
  EXPECT_FP_EQ_WITH_TOLERANCE(result, expected, 1);

  // asinpi(-0.5) = -1/6
  result = LIBC_NAMESPACE::asinpi(-0.5);
  EXPECT_FP_EQ_WITH_TOLERANCE(result, -expected, 1);
}
