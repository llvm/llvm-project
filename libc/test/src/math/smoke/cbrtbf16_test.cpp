//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Unittests for cbrtbf16

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/cbrtbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCbrtbf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcCbrtbf16Test, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::cbrtbf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cbrtbf16(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::cbrtbf16(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::cbrtbf16(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::cbrtbf16(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::cbrtbf16(neg_zero));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(1.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(1.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-1.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(-1.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(2.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(8.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-2.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(-8.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(3.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(27.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-3.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(-27.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(5.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(125.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-5.0f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(-125.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(0x1.0p42f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(0x1.0p126f)));
  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(-0x1.0p42f),
                            LIBC_NAMESPACE::cbrtbf16(bfloat16(-0x1.0p126f)));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

// the float version includes explicit FTZ-mode checks for subnormal
// outputs using hex expectations. for bfloat16, this is unnecessary.
// if x=2^e then cbrt(x)=2^(e/3). to produce a subnormal result, we would
// need e/3 < -126, i.e e<-378. Since the smallest representable exponent
// in bfloat16 is -133, no finite bfloat16 input can produce a subnormal
// cube root. therefore, explicit subnormal output checks are omitted here.

TEST_F(LlvmLibcCbrtbf16Test, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(bfloat16(0.0f), LIBC_NAMESPACE::cbrtbf16(min_denormal));
  EXPECT_FP_EQ(bfloat16(0.0f), LIBC_NAMESPACE::cbrtbf16(max_denormal));
}

TEST_F(LlvmLibcCbrtbf16Test, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(bfloat16(0.0f), LIBC_NAMESPACE::cbrtbf16(min_denormal));
  EXPECT_FP_EQ(bfloat16(0.0f), LIBC_NAMESPACE::cbrtbf16(max_denormal));
}

#endif
