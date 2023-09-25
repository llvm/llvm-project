//===-- Unittests for log1p -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log1p.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using __llvm_libc::testing::tlog;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcLog1pTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log1p(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log1p(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log1p(neg_inf), FE_INVALID);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::log1p(-2.0), FE_INVALID);
  EXPECT_FP_EQ(zero, __llvm_libc::log1p(0.0));
  EXPECT_FP_EQ(neg_zero, __llvm_libc::log1p(-0.0));
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::log1p(-1.0), FE_DIVBYZERO);
}
