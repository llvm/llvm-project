//===-- Unittests for frexp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/frexp.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/ClassificationFunctions.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;
using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

namespace mpfr = __llvm_libc::testing::mpfr;

using BitPatterns = __llvm_libc::fputil::BitPatterns<double>;
using Properties = __llvm_libc::fputil::FloatProperties<double>;

TEST(FrexpTest, SpecialNumbers) {
  int exponent;

  EXPECT_EQ(BitPatterns::aQuietNaN,
            valueAsBits(__llvm_libc::frexp(
                valueFromBits(BitPatterns::aQuietNaN), &exponent)));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::frexp(
                valueFromBits(BitPatterns::aNegativeQuietNaN), &exponent)));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::frexp(
                valueFromBits(BitPatterns::aSignallingNaN), &exponent)));
  EXPECT_EQ(
      BitPatterns::aNegativeSignallingNaN,
      valueAsBits(__llvm_libc::frexp(
          valueFromBits(BitPatterns::aNegativeSignallingNaN), &exponent)));

  EXPECT_EQ(BitPatterns::inf, valueAsBits(__llvm_libc::frexp(
                                  valueFromBits(BitPatterns::inf), &exponent)));
  EXPECT_EQ(BitPatterns::negInf,
            valueAsBits(__llvm_libc::frexp(valueFromBits(BitPatterns::negInf),
                                           &exponent)));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::frexp(valueFromBits(BitPatterns::zero),
                                           &exponent)));
  EXPECT_EQ(exponent, 0);
  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::frexp(valueFromBits(BitPatterns::negZero),
                                           &exponent)));
  EXPECT_EQ(exponent, 0);
}

TEST(FrexpTest, PowersOfTwo) {
  int exponent;

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(1.0, &exponent)));
  EXPECT_EQ(exponent, 1);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-1.0, &exponent)));
  EXPECT_EQ(exponent, 1);

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(2.0, &exponent)));
  EXPECT_EQ(exponent, 2);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-2.0, &exponent)));
  EXPECT_EQ(exponent, 2);

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(4.0, &exponent)));
  EXPECT_EQ(exponent, 3);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-4.0, &exponent)));
  EXPECT_EQ(exponent, 3);

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(8.0, &exponent)));
  EXPECT_EQ(exponent, 4);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-8.0, &exponent)));
  EXPECT_EQ(exponent, 4);

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(16.0, &exponent)));
  EXPECT_EQ(exponent, 5);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-16.0, &exponent)));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(32.0, &exponent)));
  EXPECT_EQ(exponent, 6);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-32.0, &exponent)));
  EXPECT_EQ(exponent, 6);

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::frexp(64.0, &exponent)));
  EXPECT_EQ(exponent, 7);
  EXPECT_EQ(valueAsBits(-0.5),
            valueAsBits(__llvm_libc::frexp(-64.0, &exponent)));
  EXPECT_EQ(exponent, 7);
}

TEST(FrexpTest, SomeIntegers) {
  int exponent;

  EXPECT_EQ(valueAsBits(0.75),
            valueAsBits(__llvm_libc::frexp(24.0, &exponent)));
  EXPECT_EQ(exponent, 5);
  EXPECT_EQ(valueAsBits(-0.75),
            valueAsBits(__llvm_libc::frexp(-24.0, &exponent)));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(valueAsBits(0.625),
            valueAsBits(__llvm_libc::frexp(40.0, &exponent)));
  EXPECT_EQ(exponent, 6);
  EXPECT_EQ(valueAsBits(-0.625),
            valueAsBits(__llvm_libc::frexp(-40.0, &exponent)));
  EXPECT_EQ(exponent, 6);

  EXPECT_EQ(valueAsBits(0.78125),
            valueAsBits(__llvm_libc::frexp(800.0, &exponent)));
  EXPECT_EQ(exponent, 10);
  EXPECT_EQ(valueAsBits(-0.78125),
            valueAsBits(__llvm_libc::frexp(-800.0, &exponent)));
  EXPECT_EQ(exponent, 10);
}

TEST(FrexpTest, InDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 1000001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0.0)
      continue;

    mpfr::BinaryOutput<double> result;
    result.f = __llvm_libc::frexp(x, &result.i);

    ASSERT_TRUE(__llvm_libc::fputil::abs(result.f) < 1.0);
    ASSERT_TRUE(__llvm_libc::fputil::abs(result.f) >= 0.5);
    ASSERT_MPFR_MATCH(mpfr::Operation::Frexp, x, result, 0.0);
  }
}
