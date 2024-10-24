//===-- Unittests for atan2 -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtan2Test = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAtan2Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2(aNaN, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2(1.0, aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::atan2(zero, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0, LIBC_NAMESPACE::atan2(-0.0, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(0.0, LIBC_NAMESPACE::atan2(1.0, inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0, LIBC_NAMESPACE::atan2(-1.0, inf));
}
