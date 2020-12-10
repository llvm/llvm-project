//===-- Unittests for roundf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(RoundfTest, SpecialNumbers) {
  EXPECT_FP_EQ(zero, __llvm_libc::roundf(zero));
  EXPECT_FP_EQ(negZero, __llvm_libc::roundf(negZero));

  EXPECT_FP_EQ(inf, __llvm_libc::roundf(inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::roundf(negInf));

  ASSERT_NE(isnan(nan), 0);
  ASSERT_NE(isnan(__llvm_libc::roundf(nan)), 0);
}

TEST(RoundfTest, RoundedNumbers) {
  EXPECT_FP_EQ(1.0f, __llvm_libc::roundf(1.0f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::roundf(-1.0f));
  EXPECT_FP_EQ(10.0f, __llvm_libc::roundf(10.0f));
  EXPECT_FP_EQ(-10.0f, __llvm_libc::roundf(-10.0f));
  EXPECT_FP_EQ(1234.0f, __llvm_libc::roundf(1234.0f));
  EXPECT_FP_EQ(-1234.0f, __llvm_libc::roundf(-1234.0f));
}

TEST(RoundfTest, Fractions) {
  EXPECT_FP_EQ(1.0f, __llvm_libc::roundf(0.5f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::roundf(-0.5f));
  EXPECT_FP_EQ(0.0f, __llvm_libc::roundf(0.115f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::roundf(-0.115f));
  EXPECT_FP_EQ(1.0f, __llvm_libc::roundf(0.715f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::roundf(-0.715f));
  EXPECT_FP_EQ(1.0f, __llvm_libc::roundf(1.3f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::roundf(-1.3f));
  EXPECT_FP_EQ(2.0f, __llvm_libc::roundf(1.5f));
  EXPECT_FP_EQ(-2.0f, __llvm_libc::roundf(-1.5f));
  EXPECT_FP_EQ(2.0f, __llvm_libc::roundf(1.75f));
  EXPECT_FP_EQ(-2.0f, __llvm_libc::roundf(-1.75f));
  EXPECT_FP_EQ(10.0f, __llvm_libc::roundf(10.32f));
  EXPECT_FP_EQ(-10.0f, __llvm_libc::roundf(-10.32f));
  EXPECT_FP_EQ(11.0f, __llvm_libc::roundf(10.65f));
  EXPECT_FP_EQ(-11.0f, __llvm_libc::roundf(-10.65f));
  EXPECT_FP_EQ(1234.0f, __llvm_libc::roundf(1234.38f));
  EXPECT_FP_EQ(-1234.0f, __llvm_libc::roundf(-1234.38f));
  EXPECT_FP_EQ(1235.0f, __llvm_libc::roundf(1234.96f));
  EXPECT_FP_EQ(-1235.0f, __llvm_libc::roundf(-1234.96f));
}

TEST(RoundfTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 1000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Round, x, __llvm_libc::roundf(x), 0.0);
  }
}
