//===-- Unittests for fmaxf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/fmaxf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

float nan = FPBits::buildNaN(1);
float inf = FPBits::inf();
float negInf = FPBits::negInf();

TEST(FmaxfTest, NaNArg) {
  EXPECT_EQ(inf, __llvm_libc::fmaxf(nan, inf));
  EXPECT_EQ(negInf, __llvm_libc::fmaxf(negInf, nan));
  EXPECT_EQ(0.0f, __llvm_libc::fmaxf(nan, 0.0f));
  EXPECT_EQ(-0.0f, __llvm_libc::fmaxf(-0.0f, nan));
  EXPECT_EQ(-1.2345f, __llvm_libc::fmaxf(nan, -1.2345f));
  EXPECT_EQ(1.2345f, __llvm_libc::fmaxf(1.2345f, nan));
  EXPECT_NE(isnan(__llvm_libc::fmaxf(nan, nan)), 0);
}

TEST(FmaxfTest, InfArg) {
  EXPECT_EQ(inf, __llvm_libc::fmaxf(negInf, inf));
  EXPECT_EQ(inf, __llvm_libc::fmaxf(inf, 0.0f));
  EXPECT_EQ(inf, __llvm_libc::fmaxf(-0.0f, inf));
  EXPECT_EQ(inf, __llvm_libc::fmaxf(inf, 1.2345f));
  EXPECT_EQ(inf, __llvm_libc::fmaxf(-1.2345f, inf));
}

TEST(FmaxfTest, NegInfArg) {
  EXPECT_EQ(inf, __llvm_libc::fmaxf(inf, negInf));
  EXPECT_EQ(0.0f, __llvm_libc::fmaxf(negInf, 0.0f));
  EXPECT_EQ(-0.0f, __llvm_libc::fmaxf(-0.0f, negInf));
  EXPECT_EQ(-1.2345f, __llvm_libc::fmaxf(negInf, -1.2345f));
  EXPECT_EQ(1.2345f, __llvm_libc::fmaxf(1.2345f, negInf));
}

TEST(FmaxfTest, BothZero) {
  EXPECT_EQ(0.0f, __llvm_libc::fmaxf(0.0f, 0.0f));
  EXPECT_EQ(0.0f, __llvm_libc::fmaxf(-0.0f, 0.0f));
  EXPECT_EQ(0.0f, __llvm_libc::fmaxf(0.0f, -0.0f));
  EXPECT_EQ(-0.0f, __llvm_libc::fmaxf(-0.0f, -0.0f));
}

TEST(FmaxfTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0, w = UIntType(-1); i <= count;
       ++i, v += step, w -= step) {
    float x = FPBits(v), y = FPBits(w);
    if (isnan(x) || isinf(x))
      continue;
    if (isnan(y) || isinf(y))
      continue;
    if ((x == 0) && (y == 0))
      continue;

    if (x > y) {
      ASSERT_EQ(x, __llvm_libc::fmaxf(x, y));
    } else {
      ASSERT_EQ(y, __llvm_libc::fmaxf(x, y));
    }
  }
}
