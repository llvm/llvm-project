//===-- Unittests for exp10f ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <stdint.h>

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcExp10fTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::exp10f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, __llvm_libc::exp10f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, __llvm_libc::exp10f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, __llvm_libc::exp10f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, __llvm_libc::exp10f(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(10.0f, __llvm_libc::exp10f(1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(100.0f, __llvm_libc::exp10f(2.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(1000.0f, __llvm_libc::exp10f(3.0f));
}

TEST(LlvmLibcExp10fTest, Overflow) {
  libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, __llvm_libc::exp10f(float(FPBits(0x7f7fffffU))), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, __llvm_libc::exp10f(float(FPBits(0x43000000U))), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, __llvm_libc::exp10f(float(FPBits(0x43000001U))), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}
