//===-- Unittests for exp2m1f ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/exp2m1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExp2m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

TEST_F(LlvmLibcExp2m1fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::exp2m1f(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::exp2m1f(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::exp2m1f(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::exp2m1f(0.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::exp2m1f(-0.0f));

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::exp2m1f(1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.5f, LIBC_NAMESPACE::exp2m1f(-1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(3.0f, LIBC_NAMESPACE::exp2m1f(2.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.75f, LIBC_NAMESPACE::exp2m1f(-2.0f));
}

TEST_F(LlvmLibcExp2m1fTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp2m1f(0x1.fffffep+127),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp2m1f(128.0f),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp2m1f(0x1.000002p+7),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExp2m1fTest, Underflow) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(-1.0f, LIBC_NAMESPACE::exp2m1f(-0x1.fffffep+127),
                              FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(-1.0f, LIBC_NAMESPACE::exp2m1f(-25.0f),
                              FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(-1.0f, LIBC_NAMESPACE::exp2m1f(-0x1.900002p4),
                              FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcExp2m1fTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2m1f(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2m1f(max_denormal));
}

TEST_F(LlvmLibcExp2m1fTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2m1f(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2m1f(max_denormal));
}

TEST_F(LlvmLibcExp2m1fTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2m1f(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2m1f(max_denormal));
}

#endif
