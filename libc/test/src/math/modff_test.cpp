//===-- Unittests for modff -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/modff.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/FPUtil/NearestIntegerOperations.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;
using Properties = __llvm_libc::fputil::FloatProperties<float>;

TEST(LlvmLibcModffTest, SpecialNumbers) {
  float integral;

  EXPECT_EQ(BitPatterns::aQuietNaN,
            valueAsBits(__llvm_libc::modff(
                valueFromBits(BitPatterns::aQuietNaN), &integral)));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::modff(
                valueFromBits(BitPatterns::aNegativeQuietNaN), &integral)));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::modff(
                valueFromBits(BitPatterns::aSignallingNaN), &integral)));
  EXPECT_EQ(
      BitPatterns::aNegativeSignallingNaN,
      valueAsBits(__llvm_libc::modff(
          valueFromBits(BitPatterns::aNegativeSignallingNaN), &integral)));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modff(valueFromBits(BitPatterns::inf),
                                           &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::inf);

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modff(valueFromBits(BitPatterns::negInf),
                                           &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::negInf);

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modff(valueFromBits(BitPatterns::zero),
                                           &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::zero);

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modff(valueFromBits(BitPatterns::negZero),
                                           &integral)));
  EXPECT_EQ(valueAsBits(integral), BitPatterns::negZero);
}

TEST(LlvmLibcModffTest, Integers) {
  float integral;

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modff(1.0f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(1.0f));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modff(-1.0f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-1.0f));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modff(10.0f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(10.0f));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modff(-10.0f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-10.0f));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::modff(12345.0f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(12345.0f));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::modff(-12345.0f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-12345.0f));
}

TEST(LlvmLibcModffTest, Fractions) {
  float integral;

  EXPECT_EQ(valueAsBits(0.5f),
            valueAsBits(__llvm_libc::modff(1.5f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(1.0f));

  EXPECT_EQ(valueAsBits(-0.5f),
            valueAsBits(__llvm_libc::modff(-1.5f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-1.0f));

  EXPECT_EQ(valueAsBits(0.75f),
            valueAsBits(__llvm_libc::modff(10.75f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(10.0f));

  EXPECT_EQ(valueAsBits(-0.75f),
            valueAsBits(__llvm_libc::modff(-10.75f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-10.0f));

  EXPECT_EQ(valueAsBits(0.125f),
            valueAsBits(__llvm_libc::modff(100.125f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(100.0f));

  EXPECT_EQ(valueAsBits(-0.125f),
            valueAsBits(__llvm_libc::modff(-100.125f, &integral)));
  EXPECT_EQ(valueAsBits(integral), valueAsBits(-100.0f));
}

TEST(LlvmLibcModffTest, InDoubleRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 10000000;
  constexpr BitsType step = UINT32_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x) || x == 0.0f) {
      // These conditions have been tested in other tests.
      continue;
    }

    float integral;
    float frac = __llvm_libc::modff(x, &integral);
    ASSERT_TRUE(__llvm_libc::fputil::abs(frac) < 1.0f);
    ASSERT_TRUE(__llvm_libc::fputil::trunc(x) == integral);
    ASSERT_TRUE(integral + frac == x);
  }
}
