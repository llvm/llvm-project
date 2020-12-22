//===-- Unittests for ceill -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ceill.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<long double>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(long double)

TEST(CeillTest, SpecialNumbers) {
  EXPECT_FP_EQ(zero, __llvm_libc::ceill(zero));
  EXPECT_FP_EQ(negZero, __llvm_libc::ceill(negZero));

  EXPECT_FP_EQ(inf, __llvm_libc::ceill(inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::ceill(negInf));

  EXPECT_FP_EQ(aNaN, __llvm_libc::ceill(aNaN));
}

TEST(CeillTest, RoundedNumbers) {
  EXPECT_FP_EQ(1.0l, __llvm_libc::ceill(1.0l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::ceill(-1.0l));
  EXPECT_FP_EQ(10.0l, __llvm_libc::ceill(10.0l));
  EXPECT_FP_EQ(-10.0l, __llvm_libc::ceill(-10.0l));
  EXPECT_FP_EQ(1234.0l, __llvm_libc::ceill(1234.0l));
  EXPECT_FP_EQ(-1234.0l, __llvm_libc::ceill(-1234.0l));
}

TEST(CeillTest, Fractions) {
  EXPECT_FP_EQ(1.0l, __llvm_libc::ceill(0.5l));
  EXPECT_FP_EQ(-0.0l, __llvm_libc::ceill(-0.5l));
  EXPECT_FP_EQ(1.0l, __llvm_libc::ceill(0.115l));
  EXPECT_FP_EQ(-0.0l, __llvm_libc::ceill(-0.115l));
  EXPECT_FP_EQ(1.0l, __llvm_libc::ceill(0.715l));
  EXPECT_FP_EQ(-0.0l, __llvm_libc::ceill(-0.715l));
  EXPECT_FP_EQ(2.0l, __llvm_libc::ceill(1.3l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::ceill(-1.3l));
  EXPECT_FP_EQ(2.0l, __llvm_libc::ceill(1.5l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::ceill(-1.5l));
  EXPECT_FP_EQ(2.0l, __llvm_libc::ceill(1.75l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::ceill(-1.75l));
  EXPECT_FP_EQ(11.0l, __llvm_libc::ceill(10.32l));
  EXPECT_FP_EQ(-10.0l, __llvm_libc::ceill(-10.32l));
  EXPECT_FP_EQ(11.0l, __llvm_libc::ceill(10.65l));
  EXPECT_FP_EQ(-10.0l, __llvm_libc::ceill(-10.65l));
  EXPECT_FP_EQ(1235.0l, __llvm_libc::ceill(1234.38l));
  EXPECT_FP_EQ(-1234.0l, __llvm_libc::ceill(-1234.38l));
  EXPECT_FP_EQ(1235.0l, __llvm_libc::ceill(1234.96l));
  EXPECT_FP_EQ(-1234.0l, __llvm_libc::ceill(-1234.96l));
}

TEST(CeillTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Ceil, x, __llvm_libc::ceill(x), 0.0);
  }
}
