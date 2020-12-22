//===-- Unittests for fminf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/fminf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(FminfTest, NaNArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fminf(aNaN, inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(negInf, aNaN));
  EXPECT_FP_EQ(0.0f, __llvm_libc::fminf(aNaN, 0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fminf(-0.0f, aNaN));
  EXPECT_FP_EQ(-1.2345f, __llvm_libc::fminf(aNaN, -1.2345f));
  EXPECT_FP_EQ(1.2345f, __llvm_libc::fminf(1.2345f, aNaN));
  EXPECT_FP_EQ(aNaN, __llvm_libc::fminf(aNaN, aNaN));
}

TEST(FminfTest, InfArg) {
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(negInf, inf));
  EXPECT_FP_EQ(0.0f, __llvm_libc::fminf(inf, 0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fminf(-0.0f, inf));
  EXPECT_FP_EQ(1.2345f, __llvm_libc::fminf(inf, 1.2345f));
  EXPECT_FP_EQ(-1.2345f, __llvm_libc::fminf(-1.2345f, inf));
}

TEST(FminfTest, NegInfArg) {
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(inf, negInf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(negInf, 0.0f));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(-0.0f, negInf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(negInf, -1.2345f));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminf(1.2345f, negInf));
}

TEST(FminfTest, BothZero) {
  EXPECT_FP_EQ(0.0f, __llvm_libc::fminf(0.0f, 0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fminf(-0.0f, 0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fminf(0.0f, -0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fminf(-0.0f, -0.0f));
}

TEST(FminfTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
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

    if (x < y) {
      ASSERT_FP_EQ(x, __llvm_libc::fminf(x, y));
    } else {
      ASSERT_FP_EQ(y, __llvm_libc::fminf(x, y));
    }
  }
}
