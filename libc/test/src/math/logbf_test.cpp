//===-- Unittests for logbf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/logbf.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;
using Properties = __llvm_libc::fputil::FloatProperties<float>;

TEST(LlvmLibcLogbfTest, SpecialNumbers) {
  EXPECT_EQ(
      BitPatterns::aQuietNaN,
      valueAsBits(__llvm_libc::logbf(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::logbf(
                valueFromBits(BitPatterns::aNegativeQuietNaN))));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::logbf(
                valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(BitPatterns::aNegativeSignallingNaN,
            valueAsBits(__llvm_libc::logbf(
                valueFromBits(BitPatterns::aNegativeSignallingNaN))));

  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::logbf(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(BitPatterns::inf, valueAsBits(__llvm_libc::logbf(
                                  valueFromBits(BitPatterns::negInf))));

  EXPECT_EQ(BitPatterns::negInf,
            valueAsBits(__llvm_libc::logbf(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(BitPatterns::negInf, valueAsBits(__llvm_libc::logbf(
                                     valueFromBits(BitPatterns::negZero))));
}

TEST(LlvmLibcLogbfTest, PowersOfTwo) {
  EXPECT_EQ(valueAsBits(0.0f), valueAsBits(__llvm_libc::logbf(1.0f)));
  EXPECT_EQ(valueAsBits(0.0f), valueAsBits(__llvm_libc::logbf(-1.0f)));

  EXPECT_EQ(valueAsBits(1.0f), valueAsBits(__llvm_libc::logbf(2.0f)));
  EXPECT_EQ(valueAsBits(1.0f), valueAsBits(__llvm_libc::logbf(-2.0f)));

  EXPECT_EQ(valueAsBits(2.0f), valueAsBits(__llvm_libc::logbf(4.0f)));
  EXPECT_EQ(valueAsBits(2.0f), valueAsBits(__llvm_libc::logbf(-4.0f)));

  EXPECT_EQ(valueAsBits(3.0f), valueAsBits(__llvm_libc::logbf(8.0f)));
  EXPECT_EQ(valueAsBits(3.0f), valueAsBits(__llvm_libc::logbf(-8.0f)));

  EXPECT_EQ(valueAsBits(4.0f), valueAsBits(__llvm_libc::logbf(16.0f)));
  EXPECT_EQ(valueAsBits(4.0f), valueAsBits(__llvm_libc::logbf(-16.0f)));

  EXPECT_EQ(valueAsBits(5.0f), valueAsBits(__llvm_libc::logbf(32.0f)));
  EXPECT_EQ(valueAsBits(5.0f), valueAsBits(__llvm_libc::logbf(-32.0f)));
}

TEST(LlvmLibcLogbTest, SomeIntegers) {
  EXPECT_EQ(valueAsBits(1.0f), valueAsBits(__llvm_libc::logbf(3.0f)));
  EXPECT_EQ(valueAsBits(1.0f), valueAsBits(__llvm_libc::logbf(-3.0f)));

  EXPECT_EQ(valueAsBits(2.0f), valueAsBits(__llvm_libc::logbf(7.0f)));
  EXPECT_EQ(valueAsBits(2.0f), valueAsBits(__llvm_libc::logbf(-7.0f)));

  EXPECT_EQ(valueAsBits(3.0f), valueAsBits(__llvm_libc::logbf(10.0f)));
  EXPECT_EQ(valueAsBits(3.0f), valueAsBits(__llvm_libc::logbf(-10.0f)));

  EXPECT_EQ(valueAsBits(4.0f), valueAsBits(__llvm_libc::logbf(31.0f)));
  EXPECT_EQ(valueAsBits(4.0f), valueAsBits(__llvm_libc::logbf(-31.0f)));

  EXPECT_EQ(valueAsBits(5.0f), valueAsBits(__llvm_libc::logbf(55.0f)));
  EXPECT_EQ(valueAsBits(5.0f), valueAsBits(__llvm_libc::logbf(-55.0f)));
}

TEST(LlvmLibcLogbfTest, InDoubleRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 10000000;
  constexpr BitsType step = UINT32_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x) || x == 0.0)
      continue;

    int exponent;
    __llvm_libc::fputil::frexp(x, exponent);
    ASSERT_TRUE(float(exponent) == __llvm_libc::logbf(x) + 1.0);
  }
}
