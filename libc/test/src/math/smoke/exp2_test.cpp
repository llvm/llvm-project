//===-- Unittests for 2^x -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp2.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExp2Test = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcExp2Test, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::exp2(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::exp2(aNaN));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::exp2(inf));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp2(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(zero, LIBC_NAMESPACE::exp2(-0x1.0p20),
                              FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp2(0x1.0p20), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::exp2(0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::exp2(-0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0, LIBC_NAMESPACE::exp2(1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(0.5, LIBC_NAMESPACE::exp2(-1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(4.0, LIBC_NAMESPACE::exp2(2.0));
  EXPECT_FP_EQ_ALL_ROUNDING(0.25, LIBC_NAMESPACE::exp2(-2.0));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcExp2Test, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp2(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp2(max_denormal));
}

TEST_F(LlvmLibcExp2Test, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp2(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp2(max_denormal));
}

TEST_F(LlvmLibcExp2Test, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp2(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp2(max_denormal));
}

#endif
