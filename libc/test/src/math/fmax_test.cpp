//===-- Unittests for fmax -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/fmax.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<double>;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcFmaxTest, NaNArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(aNaN, inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fmax(negInf, aNaN));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(aNaN, 0.0));
  EXPECT_FP_EQ(-0.0, __llvm_libc::fmax(-0.0, aNaN));
  EXPECT_FP_EQ(-1.2345, __llvm_libc::fmax(aNaN, -1.2345));
  EXPECT_FP_EQ(1.2345, __llvm_libc::fmax(1.2345, aNaN));
  EXPECT_FP_EQ(aNaN, __llvm_libc::fmax(aNaN, aNaN));
}

TEST(LlvmLibcFmaxTest, InfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(negInf, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(inf, 0.0));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(-0.0, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(inf, 1.2345));
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(-1.2345, inf));
}

TEST(LlvmLibcFmaxTest, NegInfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmax(inf, negInf));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(negInf, 0.0));
  EXPECT_FP_EQ(-0.0, __llvm_libc::fmax(-0.0, negInf));
  EXPECT_FP_EQ(-1.2345, __llvm_libc::fmax(negInf, -1.2345));
  EXPECT_FP_EQ(1.2345, __llvm_libc::fmax(1.2345, negInf));
}

TEST(LlvmLibcFmaxTest, BothZero) {
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(0.0, 0.0));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(-0.0, 0.0));
  EXPECT_FP_EQ(0.0, __llvm_libc::fmax(0.0, -0.0));
  EXPECT_FP_EQ(-0.0, __llvm_libc::fmax(-0.0, -0.0));
}

TEST(LlvmLibcFmaxTest, InDoubleRange) {
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
