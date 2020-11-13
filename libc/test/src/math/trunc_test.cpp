//===-- Unittests for trunc -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/trunc.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(TruncTest, SpecialNumbers) {
  EXPECT_FP_EQ(zero, __llvm_libc::trunc(zero));
  EXPECT_FP_EQ(negZero, __llvm_libc::trunc(negZero));

  EXPECT_FP_EQ(inf, __llvm_libc::trunc(inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::trunc(negInf));

  ASSERT_NE(isnan(nan), 0);
  ASSERT_NE(isnan(__llvm_libc::trunc(nan)), 0);
}

TEST(TruncTest, RoundedNumbers) {
  EXPECT_FP_EQ(1.0, __llvm_libc::trunc(1.0));
  EXPECT_FP_EQ(-1.0, __llvm_libc::trunc(-1.0));
  EXPECT_FP_EQ(10.0, __llvm_libc::trunc(10.0));
  EXPECT_FP_EQ(-10.0, __llvm_libc::trunc(-10.0));
  EXPECT_FP_EQ(1234.0, __llvm_libc::trunc(1234.0));
  EXPECT_FP_EQ(-1234.0, __llvm_libc::trunc(-1234.0));
}

TEST(TruncTest, Fractions) {
  EXPECT_FP_EQ(0.0, __llvm_libc::trunc(0.5));
  EXPECT_FP_EQ(-0.0, __llvm_libc::trunc(-0.5));
  EXPECT_FP_EQ(0.0, __llvm_libc::trunc(0.115));
  EXPECT_FP_EQ(-0.0, __llvm_libc::trunc(-0.115));
  EXPECT_FP_EQ(0.0, __llvm_libc::trunc(0.715));
  EXPECT_FP_EQ(-0.0, __llvm_libc::trunc(-0.715));
  EXPECT_FP_EQ(1.0, __llvm_libc::trunc(1.3));
  EXPECT_FP_EQ(-1.0, __llvm_libc::trunc(-1.3));
  EXPECT_FP_EQ(1.0, __llvm_libc::trunc(1.5));
  EXPECT_FP_EQ(-1.0, __llvm_libc::trunc(-1.5));
  EXPECT_FP_EQ(1.0, __llvm_libc::trunc(1.75));
  EXPECT_FP_EQ(-1.0, __llvm_libc::trunc(-1.75));
  EXPECT_FP_EQ(10.0, __llvm_libc::trunc(10.32));
  EXPECT_FP_EQ(-10.0, __llvm_libc::trunc(-10.32));
  EXPECT_FP_EQ(10.0, __llvm_libc::trunc(10.65));
  EXPECT_FP_EQ(-10.0, __llvm_libc::trunc(-10.65));
  EXPECT_FP_EQ(1234.0, __llvm_libc::trunc(1234.38));
  EXPECT_FP_EQ(-1234.0, __llvm_libc::trunc(-1234.38));
  EXPECT_FP_EQ(1234.0, __llvm_libc::trunc(1234.96));
  EXPECT_FP_EQ(-1234.0, __llvm_libc::trunc(-1234.96));
}

TEST(TruncTest, InDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Trunc, x, __llvm_libc::trunc(x), 0.0);
  }
}
