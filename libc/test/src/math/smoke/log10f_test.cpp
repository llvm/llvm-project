//===-- Unittests for log10f ----------------------------------------------===//
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
#include "src/math/log10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLog10fTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcLog10fTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log10f(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log10f(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log10f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log10f(neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log10f(0.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log10f(-0.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log10f(-1.0f), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log10f(1.0f));

  float x = 1.0f;
  for (int i = 0; i < 11; ++i, x *= 10.0f) {
    EXPECT_FP_EQ_ALL_ROUNDING(static_cast<float>(i), LIBC_NAMESPACE::log10f(x));
  }
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcLog10fTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(-0x1.66d3e7bd9a403p5f, LIBC_NAMESPACE::log10f(min_denormal));
}

TEST_F(LlvmLibcLog10fTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::log10f(min_denormal));
}

TEST_F(LlvmLibcLog10fTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::log10f(min_denormal));
}

#endif
