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

using LlvmLibcErffTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcErffTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::erff(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::erff(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::erff(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::erff(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::erff(neg_zero));
}
