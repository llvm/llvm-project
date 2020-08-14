//===-- Unittests for fmin -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/fminl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

long double nan = static_cast<long double>(FPBits::buildNaN(1));
long double inf = static_cast<long double>(FPBits::inf());
long double negInf = static_cast<long double>(FPBits::negInf());

TEST(FminlTest, NaNArg) {
  EXPECT_FP_EQ(inf, __llvm_libc::fminl(nan, inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(negInf, nan));
  EXPECT_FP_EQ(0.0L, __llvm_libc::fminl(nan, 0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fminl(-0.0L, nan));
  EXPECT_FP_EQ(-1.2345L, __llvm_libc::fminl(nan, -1.2345L));
  EXPECT_FP_EQ(1.2345L, __llvm_libc::fminl(1.2345L, nan));
  EXPECT_NE(isnan(__llvm_libc::fminl(nan, nan)), 0);
}

TEST(FminlTest, InfArg) {
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(negInf, inf));
  EXPECT_FP_EQ(0.0L, __llvm_libc::fminl(inf, 0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fminl(-0.0L, inf));
  EXPECT_FP_EQ(1.2345L, __llvm_libc::fminl(inf, 1.2345L));
  EXPECT_FP_EQ(-1.2345L, __llvm_libc::fminl(-1.2345L, inf));
}

TEST(FminlTest, NegInfArg) {
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(inf, negInf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(negInf, 0.0L));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(-0.0L, negInf));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(negInf, -1.2345L));
  EXPECT_FP_EQ(negInf, __llvm_libc::fminl(1.2345L, negInf));
}

TEST(FminlTest, BothZero) {
  EXPECT_FP_EQ(0.0L, __llvm_libc::fminl(0.0L, 0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fminl(-0.0L, 0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fminl(0.0L, -0.0L));
  EXPECT_FP_EQ(-0.0L, __llvm_libc::fminl(-0.0L, -0.0L));
}

TEST(FminlTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
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

    if (x < y) {
      ASSERT_FP_EQ(x, __llvm_libc::fminl(x, y));
    } else {
      ASSERT_FP_EQ(y, __llvm_libc::fminl(x, y));
    }
  }
}
