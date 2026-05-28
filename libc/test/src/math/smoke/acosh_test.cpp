//===-- Unittests for acosh -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/acosh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcAcoshTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAcoshTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acosh(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosh(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosh(0.0));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::acosh(1.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::acosh(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acosh(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAcoshTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  // acosh(x) for x < 1 is still a domain error in FTZ mode.
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acosh(min_denormal));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acosh(max_denormal));
}

TEST_F(LlvmLibcAcoshTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acosh(min_denormal));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acosh(max_denormal));
}

TEST_F(LlvmLibcAcoshTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acosh(min_denormal));
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::acosh(max_denormal));
}

#endif
