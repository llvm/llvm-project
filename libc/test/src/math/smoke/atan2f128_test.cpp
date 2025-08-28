//===-- Unittests for atan2f128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2f128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtan2f128Test = LIBC_NAMESPACE::testing::FPTest<float128>;

TEST_F(LlvmLibcAtan2f128Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2f128(aNaN, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2f128(1.0, aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan2f128(zero, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero,
                            LIBC_NAMESPACE::atan2f128(neg_zero, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan2f128(1.0, inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atan2f128(-1.0, inf));

  float128 x = 0x1.ffffffffffffffffffffffffffe7p1q;
  float128 y = 0x1.fffffffffffffffffffffffffff2p1q;
  float128 r = 0x1.921fb54442d18469898cc51701b3p-1q;
  EXPECT_FP_EQ(r, LIBC_NAMESPACE::atan2f128(x, y));
}
