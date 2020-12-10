//===-- Unittests for frexpf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/frexpf.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/ClassificationFunctions.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;
using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

namespace mpfr = __llvm_libc::testing::mpfr;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;
using Properties = __llvm_libc::fputil::FloatProperties<float>;

TEST(FrexpfTest, SpecialNumbers) {
  int exponent;

  EXPECT_EQ(BitPatterns::aQuietNaN,
            valueAsBits(__llvm_libc::frexpf(
                valueFromBits(BitPatterns::aQuietNaN), &exponent)));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::frexpf(
                valueFromBits(BitPatterns::aNegativeQuietNaN), &exponent)));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::frexpf(
                valueFromBits(BitPatterns::aSignallingNaN), &exponent)));
  EXPECT_EQ(
      BitPatterns::aNegativeSignallingNaN,
      valueAsBits(__llvm_libc::frexpf(
          valueFromBits(BitPatterns::aNegativeSignallingNaN), &exponent)));

  EXPECT_EQ(BitPatterns::inf, valueAsBits(__llvm_libc::frexpf(
                                  valueFromBits(BitPatterns::inf), &exponent)));
  EXPECT_EQ(BitPatterns::negInf,
            valueAsBits(__llvm_libc::frexpf(valueFromBits(BitPatterns::negInf),
                                            &exponent)));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::frexpf(valueFromBits(BitPatterns::zero),
                                            &exponent)));
  EXPECT_EQ(exponent, 0);
  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::frexpf(valueFromBits(BitPatterns::negZero),
                                            &exponent)));
  EXPECT_EQ(exponent, 0);
}

TEST(FrexpfTest, PowersOfTwo) {
  int exponent;

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(1.0f, &exponent)));
  EXPECT_EQ(exponent, 1);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-1.0f, &exponent)));
  EXPECT_EQ(exponent, 1);

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(2.0f, &exponent)));
  EXPECT_EQ(exponent, 2);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-2.0f, &exponent)));
  EXPECT_EQ(exponent, 2);

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(4.0f, &exponent)));
  EXPECT_EQ(exponent, 3);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-4.0f, &exponent)));
  EXPECT_EQ(exponent, 3);

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(8.0f, &exponent)));
  EXPECT_EQ(exponent, 4);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-8.0f, &exponent)));
  EXPECT_EQ(exponent, 4);

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(16.0f, &exponent)));
  EXPECT_EQ(exponent, 5);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-16.0f, &exponent)));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(32.0f, &exponent)));
  EXPECT_EQ(exponent, 6);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-32.0f, &exponent)));
  EXPECT_EQ(exponent, 6);

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::frexpf(64.0f, &exponent)));
  EXPECT_EQ(exponent, 7);
  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::frexpf(-64.0f, &exponent)));
  EXPECT_EQ(exponent, 7);
}

TEST(FrexpfTest, SomeIntegers) {
  int exponent;

  EXPECT_EQ(valueAsBits(0.75f),
            valueAsBits(__llvm_libc::frexpf(24.0f, &exponent)));
  EXPECT_EQ(exponent, 5);
  EXPECT_EQ(valueAsBits(-0.75f),
            valueAsBits(__llvm_libc::frexpf(-24.0f, &exponent)));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(valueAsBits(0.625f),
            valueAsBits(__llvm_libc::frexpf(40.0f, &exponent)));
  EXPECT_EQ(exponent, 6);
  EXPECT_EQ(valueAsBits(-0.625f),
            valueAsBits(__llvm_libc::frexpf(-40.0f, &exponent)));
  EXPECT_EQ(exponent, 6);

  EXPECT_EQ(valueAsBits(0.78125f),
            valueAsBits(__llvm_libc::frexpf(800.0f, &exponent)));
  EXPECT_EQ(exponent, 10);
  EXPECT_EQ(valueAsBits(-0.78125f),
            valueAsBits(__llvm_libc::frexpf(-800.0f, &exponent)));
  EXPECT_EQ(exponent, 10);
}

TEST(FrexpfTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 1000001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0.0)
      continue;

    mpfr::BinaryOutput<float> result;
    result.f = __llvm_libc::frexpf(x, &result.i);

    ASSERT_TRUE(__llvm_libc::fputil::abs(result.f) < 1.0);
    ASSERT_TRUE(__llvm_libc::fputil::abs(result.f) >= 0.5);
    ASSERT_MPFR_MATCH(mpfr::Operation::Frexp, x, result, 0.0);
  }
}
