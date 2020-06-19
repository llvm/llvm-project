//===-- Unittests for frexpl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/frexpl.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/UnitTest/Test.h"

#include <iostream>

using FPBits = __llvm_libc::fputil::FPBits<long double>;

TEST(FrexplTest, SpecialNumbers) {
  int exponent;

  EXPECT_TRUE(FPBits::inf() == __llvm_libc::frexpl(FPBits::inf(), &exponent));
  EXPECT_TRUE(FPBits::negInf() ==
              __llvm_libc::frexpl(FPBits::negInf(), &exponent));

  EXPECT_TRUE(FPBits::zero() == __llvm_libc::frexpl(FPBits::zero(), &exponent));
  EXPECT_EQ(exponent, 0);

  EXPECT_TRUE(FPBits::negZero() ==
              __llvm_libc::frexpl(FPBits::negZero(), &exponent));
  EXPECT_EQ(exponent, 0);

  EXPECT_TRUE(
      FPBits(__llvm_libc::frexpl(FPBits::buildNaN(1), &exponent)).isNaN());
}

TEST(FrexplTest, PowersOfTwo) {
  int exponent;

  EXPECT_TRUE(0.5l == __llvm_libc::frexpl(1.0l, &exponent));
  EXPECT_EQ(exponent, 1);
  EXPECT_TRUE(-0.5l == __llvm_libc::frexpl(-1.0l, &exponent));
  EXPECT_EQ(exponent, 1);

  EXPECT_TRUE(0.5l == __llvm_libc::frexpl(2.0l, &exponent));
  EXPECT_EQ(exponent, 2);
  EXPECT_TRUE(-0.5l == __llvm_libc::frexpl(-2.0l, &exponent));
  EXPECT_EQ(exponent, 2);

  EXPECT_TRUE(0.5l == __llvm_libc::frexpl(4.0l, &exponent));
  EXPECT_EQ(exponent, 3);
  EXPECT_TRUE(-0.5l == __llvm_libc::frexpl(-4.0l, &exponent));
  EXPECT_EQ(exponent, 3);

  EXPECT_TRUE(0.5l == __llvm_libc::frexpl(8.0l, &exponent));
  EXPECT_EQ(exponent, 4);
  EXPECT_TRUE(-0.5l == __llvm_libc::frexpl(-8.0l, &exponent));
  EXPECT_EQ(exponent, 4);

  EXPECT_TRUE(0.5l == __llvm_libc::frexpl(16.0l, &exponent));
  EXPECT_EQ(exponent, 5);
  EXPECT_TRUE(-0.5l == __llvm_libc::frexpl(-16.0l, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_TRUE(0.5l == __llvm_libc::frexpl(32.0l, &exponent));
  EXPECT_EQ(exponent, 6);
  EXPECT_TRUE(-0.5l == __llvm_libc::frexpl(-32.0l, &exponent));
  EXPECT_EQ(exponent, 6);
}

TEST(FrexplTest, SomeIntegers) {
  int exponent;

  EXPECT_TRUE(0.75l == __llvm_libc::frexpl(24.0l, &exponent));
  EXPECT_EQ(exponent, 5);
  EXPECT_TRUE(-0.75l == __llvm_libc::frexpl(-24.0l, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_TRUE(0.625l == __llvm_libc::frexpl(40.0l, &exponent));
  EXPECT_EQ(exponent, 6);
  EXPECT_TRUE(-0.625l == __llvm_libc::frexpl(-40.0l, &exponent));
  EXPECT_EQ(exponent, 6);

  EXPECT_TRUE(0.78125l == __llvm_libc::frexpl(800.0l, &exponent));
  EXPECT_EQ(exponent, 10);
  EXPECT_TRUE(-0.78125l == __llvm_libc::frexpl(-800.0l, &exponent));
  EXPECT_EQ(exponent, 10);
}

TEST(FrexplTest, LongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0.0l)
      continue;

    int exponent;
    long double frac = __llvm_libc::frexpl(x, &exponent);

    ASSERT_TRUE(__llvm_libc::fputil::abs(frac) < 1.0l);
    ASSERT_TRUE(__llvm_libc::fputil::abs(frac) >= 0.5l);
  }
}
