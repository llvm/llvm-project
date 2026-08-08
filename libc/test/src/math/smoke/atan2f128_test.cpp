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
  constexpr float128 PI = 0x1.921fb54442d18469898cc51701b8p+1q;
  constexpr float128 PI_OVER_2 = 0x1.921fb54442d18469898cc51701b8p+0q;
  constexpr float128 PI_OVER_4 = 0x1.921fb54442d18469898cc51701b8p-1q;
  constexpr float128 THREE_PI_OVER_4 = 0x1.2d97c7f3321d234f272993d1414ap+1q;

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan2f128(zero, inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atan2f128(neg_zero, inf));
  EXPECT_FP_EQ(PI, LIBC_NAMESPACE::atan2f128(zero, neg_zero));
  EXPECT_FP_EQ(-PI, LIBC_NAMESPACE::atan2f128(neg_zero, neg_zero));
  EXPECT_FP_EQ(PI, LIBC_NAMESPACE::atan2f128(zero, neg_inf));
  EXPECT_FP_EQ(-PI, LIBC_NAMESPACE::atan2f128(neg_zero, neg_inf));
  EXPECT_FP_EQ(PI_OVER_2, LIBC_NAMESPACE::atan2f128(inf, zero));
  EXPECT_FP_EQ(PI_OVER_2, LIBC_NAMESPACE::atan2f128(inf, neg_zero));
  EXPECT_FP_EQ(-PI_OVER_2, LIBC_NAMESPACE::atan2f128(neg_inf, zero));
  EXPECT_FP_EQ(-PI_OVER_2, LIBC_NAMESPACE::atan2f128(neg_inf, neg_zero));
  EXPECT_FP_EQ(PI_OVER_4, LIBC_NAMESPACE::atan2f128(inf, inf));
  EXPECT_FP_EQ(-PI_OVER_4, LIBC_NAMESPACE::atan2f128(neg_inf, inf));
  EXPECT_FP_EQ(THREE_PI_OVER_4, LIBC_NAMESPACE::atan2f128(inf, neg_inf));
  EXPECT_FP_EQ(-THREE_PI_OVER_4, LIBC_NAMESPACE::atan2f128(neg_inf, neg_inf));

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2f128(aNaN, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2f128(1.0, aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan2f128(1.0, inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atan2f128(-1.0, inf));

  float128 x = 0x1.ffffffffffffffffffffffffffe7p1q;
  float128 y = 0x1.fffffffffffffffffffffffffff2p1q;
  float128 r = 0x1.921fb54442d18469898cc51701b3p-1q;
  EXPECT_FP_EQ(r, LIBC_NAMESPACE::atan2f128(x, y));

  x = -0x1.f122e07fff556143p+3524q;
  y = 0x1.f122e07fff55615b75p+6316q;
  r = -0x1.ffffffffffffffe6cfcdc604fc99p-2793q;
  EXPECT_FP_EQ(r, LIBC_NAMESPACE::atan2f128(x, y));
}
