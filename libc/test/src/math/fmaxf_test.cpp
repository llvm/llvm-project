//===-- Unittests for fmaxf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/fmaxf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcFmaxfTest, NaNArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(aNaN, inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fmaxf(negInf, aNaN));
  EXPECT_FP_EQ(0.0f, __llvm_libc::fmaxf(aNaN, 0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fmaxf(-0.0f, aNaN));
  EXPECT_FP_EQ(-1.2345f, __llvm_libc::fmaxf(aNaN, -1.2345f));
  EXPECT_FP_EQ(1.2345f, __llvm_libc::fmaxf(1.2345f, aNaN));
  EXPECT_FP_EQ(aNaN, __llvm_libc::fmaxf(aNaN, aNaN));
}

TEST(LlvmLibcFmaxfTest, InfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(negInf, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(inf, 0.0f));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(-0.0f, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(inf, 1.2345f));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(-1.2345f, inf));
}

TEST(LlvmLibcFmaxfTest, NegInfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxf(inf, negInf));
  EXPECT_FP_EQ(0.0f, __llvm_libc::fmaxf(negInf, 0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fmaxf(-0.0f, negInf));
  EXPECT_FP_EQ(-1.2345f, __llvm_libc::fmaxf(negInf, -1.2345f));
  EXPECT_FP_EQ(1.2345f, __llvm_libc::fmaxf(1.2345f, negInf));
}

TEST(LlvmLibcFmaxfTest, BothZero) {
  EXPECT_FP_EQ(0.0f, __llvm_libc::fmaxf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, __llvm_libc::fmaxf(-0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, __llvm_libc::fmaxf(0.0f, -0.0f));
  EXPECT_FP_EQ(-0.0f, __llvm_libc::fmaxf(-0.0f, -0.0f));
}

TEST(LlvmLibcFmaxfTest, InFloatRange) {
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
      ASSERT_FP_EQ(x, __llvm_libc::fmaxf(x, y));
    } else {
      ASSERT_FP_EQ(y, __llvm_libc::fmaxf(x, y));
    }
  }
}
