//===-- Unittests for log10f ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log10f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcLog10fTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log10f(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log10f(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log10f(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log10f(0.0f), FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log10f(-0.0f),
                              FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log10f(-1.0f), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, __llvm_libc::log10f(1.0f));

  float x = 1.0f;
  for (int i = 0; i < 11; ++i, x *= 10.0f) {
    EXPECT_FP_EQ_ALL_ROUNDING(static_cast<float>(i), __llvm_libc::log10f(x));
  }
}
