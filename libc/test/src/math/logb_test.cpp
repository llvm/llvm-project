//===-- Unittests for logb ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/logb.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<double>;
using Properties = __llvm_libc::fputil::FloatProperties<double>;

TEST(LogbTest, SpecialNumbers) {
  EXPECT_EQ(
      BitPatterns::aQuietNaN,
      valueAsBits(__llvm_libc::logb(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::logb(
                valueFromBits(BitPatterns::aNegativeQuietNaN))));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(
                __llvm_libc::logb(valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(BitPatterns::aNegativeSignallingNaN,
            valueAsBits(__llvm_libc::logb(
                valueFromBits(BitPatterns::aNegativeSignallingNaN))));

  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::logb(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::logb(valueFromBits(BitPatterns::negInf))));

  EXPECT_EQ(BitPatterns::negInf,
            valueAsBits(__llvm_libc::logb(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(BitPatterns::negInf, valueAsBits(__llvm_libc::logb(
                                     valueFromBits(BitPatterns::negZero))));
}

TEST(LogbTest, PowersOfTwo) {
  EXPECT_EQ(valueAsBits(0.0), valueAsBits(__llvm_libc::logb(1.0)));
  EXPECT_EQ(valueAsBits(0.0), valueAsBits(__llvm_libc::logb(-1.0)));

  EXPECT_EQ(valueAsBits(1.0), valueAsBits(__llvm_libc::logb(2.0)));
  EXPECT_EQ(valueAsBits(1.0), valueAsBits(__llvm_libc::logb(-2.0)));

  EXPECT_EQ(valueAsBits(2.0), valueAsBits(__llvm_libc::logb(4.0)));
  EXPECT_EQ(valueAsBits(2.0), valueAsBits(__llvm_libc::logb(-4.0)));

  EXPECT_EQ(valueAsBits(3.0), valueAsBits(__llvm_libc::logb(8.0)));
  EXPECT_EQ(valueAsBits(3.0), valueAsBits(__llvm_libc::logb(-8.0)));

  EXPECT_EQ(valueAsBits(4.0), valueAsBits(__llvm_libc::logb(16.0)));
  EXPECT_EQ(valueAsBits(4.0), valueAsBits(__llvm_libc::logb(-16.0)));

  EXPECT_EQ(valueAsBits(5.0), valueAsBits(__llvm_libc::logb(32.0)));
  EXPECT_EQ(valueAsBits(5.0), valueAsBits(__llvm_libc::logb(-32.0)));
}

TEST(LogbTest, SomeIntegers) {
  EXPECT_EQ(valueAsBits(1.0), valueAsBits(__llvm_libc::logb(3.0)));
  EXPECT_EQ(valueAsBits(1.0), valueAsBits(__llvm_libc::logb(-3.0)));

  EXPECT_EQ(valueAsBits(2.0), valueAsBits(__llvm_libc::logb(7.0)));
  EXPECT_EQ(valueAsBits(2.0), valueAsBits(__llvm_libc::logb(-7.0)));

  EXPECT_EQ(valueAsBits(3.0), valueAsBits(__llvm_libc::logb(10.0)));
  EXPECT_EQ(valueAsBits(3.0), valueAsBits(__llvm_libc::logb(-10.0)));

  EXPECT_EQ(valueAsBits(4.0), valueAsBits(__llvm_libc::logb(31.0)));
  EXPECT_EQ(valueAsBits(4.0), valueAsBits(__llvm_libc::logb(-31.0)));

  EXPECT_EQ(valueAsBits(5.0), valueAsBits(__llvm_libc::logb(55.0)));
  EXPECT_EQ(valueAsBits(5.0), valueAsBits(__llvm_libc::logb(-55.0)));
}

TEST(LogbTest, InDoubleRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 10000000;
  constexpr BitsType step = UINT64_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = valueFromBits(v);
    if (isnan(x) || isinf(x) || x == 0.0)
      continue;

    int exponent;
    __llvm_libc::fputil::frexp(x, exponent);
    ASSERT_TRUE(double(exponent) == __llvm_libc::logb(x) + 1.0);
  }
}
