//===-- Unittests for acosf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/acosf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcAcosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAcosfTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosf(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::acosf(1.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosf(2.0f));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosf(-2.0f));
  EXPECT_MATH_ERRNO(EDOM);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAcosfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0x1.921fb6p0f, LIBC_NAMESPACE::acosf(min_denormal));
}

TEST_F(LlvmLibcAcosfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0x1.921fb6p0f, LIBC_NAMESPACE::acosf(min_denormal));
}

TEST_F(LlvmLibcAcosfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0x1.921fb6p0f, LIBC_NAMESPACE::acosf(min_denormal));
}

#endif
