//===-- Unittests for log2f -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <stdint.h>

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcLog2fTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log2f(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log2f(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log2f(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log2f(0.0f), FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log2f(-0.0f), FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log2f(-1.0f), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, __llvm_libc::log2f(1.0f));
}
