//===-- Unittests for cosf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/cosf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcCosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcCosfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::cosf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cosf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::cosf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::cosf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cosf(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cosf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcCosfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(max_denormal));
}

TEST_F(LlvmLibcCosfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(max_denormal));
}

TEST_F(LlvmLibcCosfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(max_denormal));
}

#endif
