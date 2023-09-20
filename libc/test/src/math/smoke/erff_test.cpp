//===-- Unittests for erff ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/erff.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using __llvm_libc::testing::tlog;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcErffTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::erff(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, __llvm_libc::erff(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, __llvm_libc::erff(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, __llvm_libc::erff(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, __llvm_libc::erff(neg_zero));
}
