//===-- Unittests for logf-----------------------------------------------===//
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
#include "src/math/logf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLogfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcLogfTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::logf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::logf(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::logf(inf));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::logf(neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::logf(0.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::logf(-0.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::logf(-1.0f), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::logf(1.0f));
}
#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcLogfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(-0x1.9d1d9fccf477p6f, LIBC_NAMESPACE::logf(min_denormal));
}

TEST_F(LlvmLibcLogfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::logf(min_denormal));
}

TEST_F(LlvmLibcLogfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::logf(min_denormal));
}

#endif
