//===-- Unittests for modfl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/modfl.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/NearestIntegerOperations.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<long double>;

TEST(modflTest, SpecialNumbers) {
  long double integral;

  EXPECT_TRUE(FPBits::zero() == __llvm_libc::modfl(FPBits::inf(), &integral));
  EXPECT_TRUE(FPBits::inf() == integral);

  EXPECT_TRUE(FPBits::negZero() ==
              __llvm_libc::modfl(FPBits::negInf(), &integral));
  EXPECT_TRUE(FPBits::negInf() == integral);

  EXPECT_TRUE(FPBits::zero() == __llvm_libc::modfl(FPBits::zero(), &integral));
  EXPECT_TRUE(integral == 0.0l);

  EXPECT_TRUE(FPBits::negZero() ==
              __llvm_libc::modfl(FPBits::negZero(), &integral));
  EXPECT_TRUE(integral == 0.0l);

  EXPECT_TRUE(
      FPBits(__llvm_libc::modfl(FPBits::buildNaN(1), &integral)).isNaN());
}

TEST(modflTest, Integers) {
  long double integral;

  EXPECT_TRUE(FPBits::zero() == __llvm_libc::modfl(1.0l, &integral));
  EXPECT_TRUE(integral == 1.0l);

  EXPECT_TRUE(FPBits::negZero() == __llvm_libc::modfl(-1.0l, &integral));
  EXPECT_TRUE(integral == -1.0l);

  EXPECT_TRUE(FPBits::zero() == __llvm_libc::modfl(10.0l, &integral));
  EXPECT_TRUE(integral == 10.0l);

  EXPECT_TRUE(FPBits::negZero() == __llvm_libc::modfl(-10.0l, &integral));
  EXPECT_TRUE(integral == -10.0l);

  EXPECT_TRUE(FPBits::zero() == __llvm_libc::modfl(12345.0l, &integral));
  EXPECT_TRUE(integral == 12345.0l);

  EXPECT_TRUE(FPBits::negZero() == __llvm_libc::modfl(-12345.0l, &integral));
  EXPECT_TRUE(integral == -12345.0l);
}

TEST(ModfTest, Fractions) {
  long double integral;

  EXPECT_TRUE(0.5l == __llvm_libc::modfl(1.5l, &integral));
  EXPECT_TRUE(integral == 1.0l);

  EXPECT_TRUE(-0.5l == __llvm_libc::modfl(-1.5l, &integral));
  EXPECT_TRUE(integral == -1.0l);

  EXPECT_TRUE(0.75l == __llvm_libc::modfl(10.75l, &integral));
  EXPECT_TRUE(integral == 10.0l);

  EXPECT_TRUE(-0.75l == __llvm_libc::modfl(-10.75l, &integral));
  EXPECT_TRUE(integral == -10.0l);

  EXPECT_TRUE(0.125l == __llvm_libc::modfl(100.125l, &integral));
  EXPECT_TRUE(integral == 100.0l);

  EXPECT_TRUE(-0.125l == __llvm_libc::modfl(-100.125l, &integral));
  EXPECT_TRUE(integral == -100.0l);
}

TEST(ModflTest, LongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0.0l)
      continue;

    long double integral;
    long double frac = __llvm_libc::modfl(x, &integral);
    ASSERT_TRUE(__llvm_libc::fputil::abs(frac) < 1.0l);
    ASSERT_TRUE(__llvm_libc::fputil::trunc(x) == integral);
    ASSERT_TRUE(integral + frac == x);
  }
}
