//===-- Unittests for acoshf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/acoshf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcAcoshfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAcoshfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acoshf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf(0.0f));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::acoshf(1.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::acoshf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAcoshfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acoshf(min_denormal));
}

TEST_F(LlvmLibcAcoshfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acoshf(min_denormal));
}

TEST_F(LlvmLibcAcoshfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acoshf(min_denormal));
}

#endif
