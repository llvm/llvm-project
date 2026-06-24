//===-- Unittests for sinh ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sinh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSinhTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcSinhTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::sinh(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::sinh(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::sinh(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::sinh(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::sinh(0.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::sinh(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcSinhTest, Overflow) {
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::sinh(0x1.0p10), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::sinh(-0x1.0p10),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcSinhTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinh(neg_min_denormal));
}

TEST_F(LlvmLibcSinhTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinh(neg_min_denormal));
}

TEST_F(LlvmLibcSinhTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::sinh(neg_min_denormal));
}

#endif
