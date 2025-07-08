//===-- Unittests for log2f -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/log2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcLog2fTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcLog2fTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::log2f(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log2f(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log2f(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log2f(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log2f(0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log2f(-0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log2f(-1.0f), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log2f(1.0f));
}
#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcLog2fTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(-149.0f, LIBC_NAMESPACE::log2f(min_denormal));
}

TEST_F(LlvmLibcLog2fTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::log2f(min_denormal));
}

TEST_F(LlvmLibcLog2fTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(FPBits::inf(Sign::NEG).get_val(),
               LIBC_NAMESPACE::log2f(min_denormal));
}

#endif
