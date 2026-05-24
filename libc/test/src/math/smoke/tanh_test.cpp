//===-- Unittests for tanh ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/tanh.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanhTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcTanhTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tanh(sNaN), FE_INVALID);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tanh(aNaN));

  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::tanh(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0, LIBC_NAMESPACE::tanh(neg_inf));

  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::tanh(0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::tanh(neg_zero));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcTanhTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tanh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::tanh(neg_min_denormal));
}

TEST_F(LlvmLibcTanhTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tanh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::tanh(neg_min_denormal));
}

TEST_F(LlvmLibcTanhTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  const double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::tanh(min_denormal));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::tanh(neg_min_denormal));
}

#endif
