//===-- Unittests for log1pf ----------------------------------------------===//
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
#include "src/math/log1pf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLog1pfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcLog1pfTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log1pf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log1pf(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log1pf(inf));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log1pf(neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::log1pf(0.0f));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::log1pf(-0.0f));
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log1pf(-1.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcLog1pfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::log1pf(min_denormal));
}

TEST_F(LlvmLibcLog1pfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::log1pf(min_denormal));
}

TEST_F(LlvmLibcLog1pfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::log1pf(min_denormal));
}

#endif
