//===-- Unittests for fmin -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/fmin.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;

double nan = static_cast<double>(FPBits::buildNaN(1));
double inf = static_cast<double>(FPBits::inf());
double negInf = static_cast<double>(FPBits::negInf());

TEST(FminTest, NaNArg) {
  EXPECT_EQ(inf, __llvm_libc::fmin(nan, inf));
  EXPECT_EQ(negInf, __llvm_libc::fmin(negInf, nan));
  EXPECT_EQ(0.0, __llvm_libc::fmin(nan, 0.0));
  EXPECT_EQ(-0.0, __llvm_libc::fmin(-0.0, nan));
  EXPECT_EQ(-1.2345, __llvm_libc::fmin(nan, -1.2345));
  EXPECT_EQ(1.2345, __llvm_libc::fmin(1.2345, nan));
  EXPECT_NE(isnan(__llvm_libc::fmin(nan, nan)), 0);
}

TEST(FminTest, InfArg) {
  EXPECT_EQ(negInf, __llvm_libc::fmin(negInf, inf));
  EXPECT_EQ(0.0, __llvm_libc::fmin(inf, 0.0));
  EXPECT_EQ(-0.0, __llvm_libc::fmin(-0.0, inf));
  EXPECT_EQ(1.2345, __llvm_libc::fmin(inf, 1.2345));
  EXPECT_EQ(-1.2345, __llvm_libc::fmin(-1.2345, inf));
}

TEST(FminTest, NegInfArg) {
  EXPECT_EQ(negInf, __llvm_libc::fmin(inf, negInf));
  EXPECT_EQ(negInf, __llvm_libc::fmin(negInf, 0.0));
  EXPECT_EQ(negInf, __llvm_libc::fmin(-0.0, negInf));
  EXPECT_EQ(negInf, __llvm_libc::fmin(negInf, -1.2345));
  EXPECT_EQ(negInf, __llvm_libc::fmin(1.2345, negInf));
}

TEST(FminTest, BothZero) {
  EXPECT_EQ(0.0, __llvm_libc::fmin(0.0, 0.0));
  EXPECT_EQ(-0.0, __llvm_libc::fmin(-0.0, 0.0));
  EXPECT_EQ(-0.0, __llvm_libc::fmin(0.0, -0.0));
  EXPECT_EQ(-0.0, __llvm_libc::fmin(-0.0, -0.0));
}

TEST(FminTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0, w = UIntType(-1); i <= count;
       ++i, v += step, w -= step) {
    double x = FPBits(v), y = FPBits(w);
    if (isnan(x) || isinf(x))
      continue;
    if (isnan(y) || isinf(y))
      continue;
    if ((x == 0) && (y == 0))
      continue;

    if (x < y) {
      ASSERT_EQ(x, __llvm_libc::fmin(x, y));
    } else {
      ASSERT_EQ(y, __llvm_libc::fmin(x, y));
    }
  }
}
