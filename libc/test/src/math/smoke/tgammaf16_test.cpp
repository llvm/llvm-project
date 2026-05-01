//===-- Unittests for tgammaf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/tgammaf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTgammaf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcTgammaf16Test, SpecialNumbers) {
  // sNaN -> qNaN + FE_INVALID
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgammaf16(sNaN),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  // qNaN -> qNaN
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tgammaf16(aNaN));

  // +Inf -> +Inf
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::tgammaf16(inf));

  // -Inf -> NaN + FE_INVALID (domain error)
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgammaf16(neg_inf),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  // +0 -> +Inf + FE_DIVBYZERO (pole error)
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::tgammaf16(zero),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  // -0 -> -Inf + FE_DIVBYZERO (pole error)
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::tgammaf16(neg_zero),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcTgammaf16Test, PositiveIntegers) {
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f16, LIBC_NAMESPACE::tgammaf16(1.0f16));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f16, LIBC_NAMESPACE::tgammaf16(2.0f16));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0f16, LIBC_NAMESPACE::tgammaf16(3.0f16));
  EXPECT_FP_EQ_ALL_ROUNDING(6.0f16, LIBC_NAMESPACE::tgammaf16(4.0f16));
  EXPECT_FP_EQ_ALL_ROUNDING(24.0f16, LIBC_NAMESPACE::tgammaf16(5.0f16));
}

TEST_F(LlvmLibcTgammaf16Test, NegativeIntegers) {
  // tgamma of negative integer -> NaN + FE_INVALID (domain error)
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgammaf16(-1.0f16),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgammaf16(-2.0f16),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcTgammaf16Test, Overflow) {
  // x >= 9.2265625 overflows float16
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::tgammaf16(9.25f16),
                              FE_OVERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcTgammaf16Test, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::tgammaf16(min_denormal));
}

TEST_F(LlvmLibcTgammaf16Test, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::tgammaf16(min_denormal));
}

TEST_F(LlvmLibcTgammaf16Test, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::tgammaf16(min_denormal));
}

#endif
