//===-- Unittests for logbl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/logbl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

TEST(logblTest, SpecialNumbers) {
  EXPECT_TRUE(FPBits::inf() == __llvm_libc::logbl(FPBits::inf()));
  EXPECT_TRUE(FPBits::inf() == __llvm_libc::logbl(FPBits::negInf()));

  EXPECT_TRUE(FPBits::negInf() == __llvm_libc::logbl(FPBits::zero()));
  EXPECT_TRUE(FPBits::negInf() == __llvm_libc::logbl(FPBits::negZero()));

  EXPECT_TRUE(FPBits(__llvm_libc::logbl(FPBits::buildNaN(1))).isNaN());
}

TEST(logblTest, PowersOfTwo) {
  EXPECT_TRUE(0.0l == __llvm_libc::logbl(1.0l));
  EXPECT_TRUE(0.0l == __llvm_libc::logbl(-1.0l));

  EXPECT_TRUE(1.0l == __llvm_libc::logbl(2.0l));
  EXPECT_TRUE(1.0l == __llvm_libc::logbl(-2.0l));

  EXPECT_TRUE(2.0l == __llvm_libc::logbl(4.0l));
  EXPECT_TRUE(2.0l == __llvm_libc::logbl(-4.0l));

  EXPECT_TRUE(3.0l == __llvm_libc::logbl(8.0l));
  EXPECT_TRUE(3.0l == __llvm_libc::logbl(-8.0l));

  EXPECT_TRUE(4.0l == __llvm_libc::logbl(16.0l));
  EXPECT_TRUE(4.0l == __llvm_libc::logbl(-16.0l));

  EXPECT_TRUE(5.0l == __llvm_libc::logbl(32.0l));
  EXPECT_TRUE(5.0l == __llvm_libc::logbl(-32.0l));
}

TEST(LogbTest, SomeIntegers) {
  EXPECT_TRUE(1.0l == __llvm_libc::logbl(3.0l));
  EXPECT_TRUE(1.0l == __llvm_libc::logbl(-3.0l));

  EXPECT_TRUE(2.0l == __llvm_libc::logbl(7.0l));
  EXPECT_TRUE(2.0l == __llvm_libc::logbl(-7.0l));

  EXPECT_TRUE(3.0l == __llvm_libc::logbl(10.0l));
  EXPECT_TRUE(3.0l == __llvm_libc::logbl(-10.0l));

  EXPECT_TRUE(4.0l == __llvm_libc::logbl(31.0l));
  EXPECT_TRUE(4.0l == __llvm_libc::logbl(-31.0l));

  EXPECT_TRUE(5.0l == __llvm_libc::logbl(55.0l));
  EXPECT_TRUE(5.0l == __llvm_libc::logbl(-55.0l));
}

TEST(LogblTest, LongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0.0l)
      continue;

    int exponent;
    __llvm_libc::fputil::frexp(x, exponent);
    ASSERT_TRUE((long double)(exponent) == __llvm_libc::logbl(x) + 1.0l);
  }
}
