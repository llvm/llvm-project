//===-- Unittests for atanh -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atanh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtanhTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAtanhTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atanh(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanh(aNaN));
  EXPECT_MATH_ERRNO(0);

  // atanh(+/-1) = +/-inf with ERANGE/FE_DIVBYZERO.
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::atanh(1.0), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::atanh(-1.0),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  // atanh(x) for |x| > 1 is a domain error.
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atanh(2.0), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atanh(inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::atanh(0.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atanh(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAtanhTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::atanh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atanh(neg_min_denormal));
}

TEST_F(LlvmLibcAtanhTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::atanh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atanh(neg_min_denormal));
}

TEST_F(LlvmLibcAtanhTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::atanh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atanh(neg_min_denormal));
}

#endif
