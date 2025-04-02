//===-- Unittests for atan ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtanTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAtanTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan(aNaN));
  // atan(sNaN) = aNaN.
  EXPECT_EQ(FPBits(aNaN).uintval(),
            FPBits(LIBC_NAMESPACE::atan(sNaN)).uintval());
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atan(neg_zero));
  // atan(+-Inf) = +- pi/2.
  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::atan(inf));
  EXPECT_FP_EQ(-0x1.921fb54442d18p0, LIBC_NAMESPACE::atan(neg_inf));
  // atan(+-1) = +- pi/4.
  EXPECT_FP_EQ(0x1.921fb54442d18p-1, LIBC_NAMESPACE::atan(1.0));
  EXPECT_FP_EQ(-0x1.921fb54442d18p-1, LIBC_NAMESPACE::atan(-1.0));
}
