//===-- Unittests for 10^x ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/exp10.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcExp10Test = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcExp10Test, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::exp10(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::exp10(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::exp10(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp10(neg_inf));
  EXPECT_FP_EQ_WITH_EXCEPTION(zero, LIBC_NAMESPACE::exp10(-0x1.0p20),
                              FE_UNDERFLOW);
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp10(0x1.0p20),
                              FE_OVERFLOW);
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::exp10(0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::exp10(-0.0));

  EXPECT_FP_EQ_ALL_ROUNDING(10.0, LIBC_NAMESPACE::exp10(1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(100.0, LIBC_NAMESPACE::exp10(2.0));
  EXPECT_FP_EQ_ALL_ROUNDING(1000.0, LIBC_NAMESPACE::exp10(3.0));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcExp10Test, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp10(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp10(max_denormal));
}

TEST_F(LlvmLibcExp10Test, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp10(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp10(max_denormal));
}

TEST_F(LlvmLibcExp10Test, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp10(min_denormal));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::exp10(max_denormal));
}

#endif
