//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for log2.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log2.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcLog2Test = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcLog2Test, SpecialNumbers) {
  const double neg_sNaN = LIBC_NAMESPACE::testing::opaque(
      FPBits::signaling_nan(Sign::NEG).get_val());
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log2(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);
  EXPECT_TRUE(FPBits(LIBC_NAMESPACE::log2(sNaN)).is_quiet_nan());

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log2(neg_sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);
  EXPECT_TRUE(FPBits(LIBC_NAMESPACE::log2(neg_sNaN)).is_quiet_nan());

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log2(aNaN));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(neg_aNaN, LIBC_NAMESPACE::log2(neg_aNaN));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log2(inf));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log2(neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log2(0.0), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log2(-0.0),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log2(-1.0), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log2(1.0));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcLog2Test, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(-1074.0, LIBC_NAMESPACE::log2(min_denormal));
}

TEST_F(LlvmLibcLog2Test, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::log2(min_denormal));
}

TEST_F(LlvmLibcLog2Test, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::log2(min_denormal));
}

#endif
