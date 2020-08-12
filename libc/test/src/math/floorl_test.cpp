//===-- Unittests for floorl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/floorl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

namespace mpfr = __llvm_libc::testing::mpfr;

static const long double zero = FPBits::zero();
static const long double negZero = FPBits::negZero();
static const long double nan = FPBits::buildNaN(1);
static const long double inf = FPBits::inf();
static const long double negInf = FPBits::negInf();

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 0,
                                           0};
TEST(FloorlTest, SpecialNumbers) {
  EXPECT_FP_EQ(zero, __llvm_libc::floorl(zero));
  EXPECT_FP_EQ(negZero, __llvm_libc::floorl(negZero));

  EXPECT_FP_EQ(inf, __llvm_libc::floorl(inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::floorl(negInf));

  ASSERT_NE(isnan(nan), 0);
  ASSERT_NE(isnan(__llvm_libc::floorl(nan)), 0);
}

TEST(FloorlTest, RoundedNumbers) {
  EXPECT_FP_EQ(1.0l, __llvm_libc::floorl(1.0l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::floorl(-1.0l));
  EXPECT_FP_EQ(10.0l, __llvm_libc::floorl(10.0l));
  EXPECT_FP_EQ(-10.0l, __llvm_libc::floorl(-10.0l));
  EXPECT_FP_EQ(1234.0l, __llvm_libc::floorl(1234.0l));
  EXPECT_FP_EQ(-1234.0l, __llvm_libc::floorl(-1234.0l));
}

TEST(FloorlTest, Fractions) {
  EXPECT_FP_EQ(0.0l, __llvm_libc::floorl(0.5l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::floorl(-0.5l));
  EXPECT_FP_EQ(0.0l, __llvm_libc::floorl(0.115l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::floorl(-0.115l));
  EXPECT_FP_EQ(0.0l, __llvm_libc::floorl(0.715l));
  EXPECT_FP_EQ(-1.0l, __llvm_libc::floorl(-0.715l));
  EXPECT_FP_EQ(1.0l, __llvm_libc::floorl(1.3l));
  EXPECT_FP_EQ(-2.0l, __llvm_libc::floorl(-1.3l));
  EXPECT_FP_EQ(1.0l, __llvm_libc::floorl(1.5l));
  EXPECT_FP_EQ(-2.0l, __llvm_libc::floorl(-1.5l));
  EXPECT_FP_EQ(1.0l, __llvm_libc::floorl(1.75l));
  EXPECT_FP_EQ(-2.0l, __llvm_libc::floorl(-1.75l));
  EXPECT_FP_EQ(10.0l, __llvm_libc::floorl(10.32l));
  EXPECT_FP_EQ(-11.0l, __llvm_libc::floorl(-10.32l));
  EXPECT_FP_EQ(10.0l, __llvm_libc::floorl(10.65l));
  EXPECT_FP_EQ(-11.0l, __llvm_libc::floorl(-10.65l));
  EXPECT_FP_EQ(1234.0l, __llvm_libc::floorl(1234.38l));
  EXPECT_FP_EQ(-1235.0l, __llvm_libc::floorl(-1234.38l));
  EXPECT_FP_EQ(1234.0l, __llvm_libc::floorl(1234.96l));
  EXPECT_FP_EQ(-1235.0l, __llvm_libc::floorl(-1234.96l));
}

TEST(FloorlTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Floor, x, __llvm_libc::floorl(x),
                      tolerance);
  }
}
