//===-- Unittests for atan2f128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float128.h"
#include "src/__support/integer_literals.h"
#include "src/math/atan2f128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

using LIBC_NAMESPACE::operator""_u128;

using LlvmLibcAtan2f128Test = LIBC_NAMESPACE::testing::FPTest<float128>;

TEST_F(LlvmLibcAtan2f128Test, SpecialNumbers) {
  const float128 PI = FPBits(0x4000921f'b54442d1'8469898c'c51701b8_u128)
                          .get_val(); // 0x1.921fb54442d18469898cc51701b8p+1q
  const float128 PI_OVER_2 =
      FPBits(0x3fff921f'b54442d1'8469898c'c51701b8_u128)
          .get_val(); // 0x1.921fb54442d18469898cc51701b8p+0q
  const float128 PI_OVER_4 =
      FPBits(0x3ffe921f'b54442d1'8469898c'c51701b8_u128)
          .get_val(); // 0x1.921fb54442d18469898cc51701b8p-1q
  const float128 THREE_PI_OVER_4 =
      FPBits(0x40002d97'c7f3321d'234f2729'93d1414a_u128)
          .get_val(); // 0x1.2d97c7f3321d234f272993d1414ap+1q

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
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN,
                            LIBC_NAMESPACE::atan2f128(float128(1.0), aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(zero,
                            LIBC_NAMESPACE::atan2f128(float128(1.0), inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero,
                            LIBC_NAMESPACE::atan2f128(float128(-1.0), inf));

  float128 x = FPBits(0x4000ffff'ffffffff'ffffffff'ffffffe7_u128)
                   .get_val(); // 0x1.ffffffffffffffffffffffffffe7p1q
  float128 y = FPBits(0x4000ffff'ffffffff'ffffffff'fffffff2_u128)
                   .get_val(); // 0x1.fffffffffffffffffffffffffff2p1q
  float128 r = FPBits(0x3ffe921f'b54442d1'8469898c'c51701b3_u128)
                   .get_val(); // 0x1.921fb54442d18469898cc51701b3p-1q
  EXPECT_FP_EQ(r, LIBC_NAMESPACE::atan2f128(x, y));

  x = FPBits(0xcdc3f122'e07fff55'61430000'00000000_u128)
          .get_val(); // -0x1.f122e07fff556143p+3524q
  y = FPBits(0x58abf122'e07fff55'615b7500'00000000_u128)
          .get_val(); // 0x1.f122e07fff55615b75p+6316q
  r = FPBits(0xb516ffff'ffffffff'ffe6cfcd'c604fc99_u128)
          .get_val(); // -0x1.ffffffffffffffe6cfcdc604fc99p-2793q
  EXPECT_FP_EQ(r, LIBC_NAMESPACE::atan2f128(x, y));
}
