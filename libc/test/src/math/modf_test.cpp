//===-- Unittests for modf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/modf.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/FPUtil/NearestIntegerOperations.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<double>;
using Properties = __llvm_libc::fputil::FloatProperties<double>;

TEST(ModfTest, SpecialNumbers) {
  double integral;

  EXPECT_EQ(BitPatterns::aQuietNaN,
            valueAsBits(__llvm_libc::modf(valueFromBits(BitPatterns::aQuietNaN),
                                          &integral)));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::modf(
                valueFromBits(BitPatterns::aNegativeQuietNaN), &integral)));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::modf(
                valueFromBits(BitPatterns::aSignallingNaN), &integral)));
  EXPECT_EQ(
      BitPatterns::aNegativeSignallingNaN,
      valueAsBits(__llvm_libc::modf(
          valueFromBits(BitPatterns::aNegativeSignallingNaN), &integral)));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(
                __llvm_libc::modf(valueFromBits(BitPatterns::inf), &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::inf);

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modf(valueFromBits(BitPatterns::negInf),
                                          &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::negInf);

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modf(valueFromBits(BitPatterns::zero),
                                          &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::zero);

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modf(valueFromBits(BitPatterns::negZero),
                                          &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::negZero);
}

TEST(ModfTest, Integers) {
  double integral;

  EXPECT_EQ(BitPatterns::zero, valueAsBits(__llvm_libc::modf(1.0, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(1.0));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modf(-1.0, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-1.0));

  EXPECT_EQ(BitPatterns::zero, valueAsBits(__llvm_libc::modf(10.0, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(10.0));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modf(-10.0, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-10.0));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modf(12345.0, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(12345.0));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modf(-12345.0, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-12345.0));
}

TEST(ModfTest, Fractions) {
  double integral;

  EXPECT_EQ(valueAsBits(0.5), valueAsBits(__llvm_libc::modf(1.5, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(1.0));

  EXPECT_EQ(valueAsBits(-0.5), valueAsBits(__llvm_libc::modf(-1.5, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-1.0));

  EXPECT_EQ(valueAsBits(0.75),
            valueAsBits(__llvm_libc::modf(10.75, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(10.0));

  EXPECT_EQ(valueAsBits(-0.75),
            valueAsBits(__llvm_libc::modf(-10.75, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-10.0));

  EXPECT_EQ(valueAsBits(0.125),
            valueAsBits(__llvm_libc::modf(100.125, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(100.0));

  EXPECT_EQ(valueAsBits(-0.125),
            valueAsBits(__llvm_libc::modf(-100.125, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-100.0));
}

TEST(ModfTest, InDoubleRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 10000000;
  constexpr BitsType step = UINT64_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = valueFromBits(v);
    if (isnan(x) || isinf(x) || x == 0.0) {
      // These conditions have been tested in other tests.
      continue;
    }

    double integral;
    double frac = __llvm_libc::modf(x, &integral);
    ASSERT_TRUE(__llvm_libc::fputil::abs(frac) < 1.0);
    ASSERT_TRUE(__llvm_libc::fputil::trunc(x) == integral);
    ASSERT_TRUE(integral + frac == x);
  }
}
