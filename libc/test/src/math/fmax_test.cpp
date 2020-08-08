//===-- Unittests for fmax -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/fmax.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;

double nan = FPBits::buildNaN(1);
double inf = FPBits::inf();
double negInf = FPBits::negInf();

TEST(FmaxTest, NaNArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(nan, inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fmax(negInf, nan));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(nan, 0.0));
  EXPECT_FP_EQ(-0.0, __llvm_libc::fmax(-0.0, nan));
  EXPECT_FP_EQ(-1.2345, __llvm_libc::fmax(nan, -1.2345));
  EXPECT_FP_EQ(1.2345, __llvm_libc::fmax(1.2345, nan));
  EXPECT_NE(isnan(__llvm_libc::fmax(nan, nan)), 0);
}

TEST(FmaxTest, InfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(negInf, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(inf, 0.0));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(-0.0, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(inf, 1.2345));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(-1.2345, inf));
}

TEST(FmaxTest, NegInfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(inf, negInf));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(negInf, 0.0));
  EXPECT_FP_EQ(-0.0, __llvm_libc::fmax(-0.0, negInf));
  EXPECT_FP_EQ(-1.2345, __llvm_libc::fmax(negInf, -1.2345));
  EXPECT_FP_EQ(1.2345, __llvm_libc::fmax(1.2345, negInf));
}

TEST(FmaxTest, BothZero) {
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(0.0, 0.0));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(-0.0, 0.0));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(0.0, -0.0));
  EXPECT_FP_EQ(-0.0, __llvm_libc::fmax(-0.0, -0.0));
}

TEST(FmaxTest, InDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000001;
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

    if (x > y) {
      EXPECT_FP_EQ(x, __llvm_libc::fmax(x, y));
    } else {
      EXPECT_FP_EQ(y, __llvm_libc::fmax(x, y));
    }
  }
}
