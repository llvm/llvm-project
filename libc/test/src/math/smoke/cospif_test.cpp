//===-- Unittests for cospif ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/math/cospif.h"
#include "test/UnitTest/FPMatcher.h"

#include <stdint.h>

using LlvmLibcCospifTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcCospifTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::cospif(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospif(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospif(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospif(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcCospifTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(max_denormal));
}

TEST_F(LlvmLibcCospifTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(max_denormal));
}

TEST_F(LlvmLibcCospifTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cospif(max_denormal));
}

#endif
