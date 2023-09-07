//===-- Unittests for 2^x -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp2.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using __llvm_libc::testing::tlog;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcExp2Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::exp2(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::exp2(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, __llvm_libc::exp2(neg_inf));
  EXPECT_FP_EQ_WITH_EXCEPTION(zero, __llvm_libc::exp2(-0x1.0p20), FE_UNDERFLOW);
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, __llvm_libc::exp2(0x1.0p20), FE_OVERFLOW);
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, __llvm_libc::exp2(0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, __llvm_libc::exp2(-0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0, __llvm_libc::exp2(1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(0.5, __llvm_libc::exp2(-1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(4.0, __llvm_libc::exp2(2.0));
  EXPECT_FP_EQ_ALL_ROUNDING(0.25, __llvm_libc::exp2(-2.0));
}
