//===-- Unittests for fmaxl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/fmaxl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<long double>;

DECLARE_SPECIAL_CONSTANTS(long double)

TEST(LlvmLibcFmaxlTest, NaNArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(aNaN, inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fmaxl(negInf, aNaN));
  EXPECT_FP_EQ(0.0L, __llvm_libc::fmaxl(aNaN, 0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fmaxl(-0.0L, aNaN));
  EXPECT_FP_EQ(-1.2345L, __llvm_libc::fmaxl(aNaN, -1.2345L));
  EXPECT_FP_EQ(1.2345L, __llvm_libc::fmaxl(1.2345L, aNaN));
  EXPECT_FP_EQ(aNaN, __llvm_libc::fmaxl(aNaN, aNaN));
}

TEST(LlvmLibcFmaxlTest, InfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(negInf, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(inf, 0.0L));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(-0.0L, inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(inf, 1.2345L));
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(-1.2345L, inf));
}

TEST(LlvmLibcFmaxlTest, NegInfArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fmaxl(inf, negInf));
  EXPECT_FP_EQ(0.0L, __llvm_libc::fmaxl(negInf, 0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fmaxl(-0.0L, negInf));
  EXPECT_FP_EQ(-1.2345L, __llvm_libc::fmaxl(negInf, -1.2345L));
  EXPECT_FP_EQ(1.2345L, __llvm_libc::fmaxl(1.2345L, negInf));
}

TEST(LlvmLibcFmaxlTest, BothZero) {
  EXPECT_FP_EQ(0.0L, __llvm_libc::fmaxl(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0L, __llvm_libc::fmaxl(-0.0L, 0.0L));
  EXPECT_FP_EQ(0.0L, __llvm_libc::fmaxl(0.0L, -0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fmaxl(-0.0L, -0.0L));
}

TEST(LlvmLibcFmaxlTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0, w = UIntType(-1); i <= count;
       ++i, v += step, w -= step) {
    long double x = FPBits(v), y = FPBits(w);
    if (isnan(x) || isinf(x))
      continue;
    if (isnan(y) || isinf(y))
      continue;
    if ((x == 0) && (y == 0))
      continue;

    if (x > y) {
      ASSERT_FP_EQ(x, __llvm_libc::fmaxl(x, y));
    } else {
      ASSERT_FP_EQ(y, __llvm_libc::fmaxl(x, y));
    }
  }
}
