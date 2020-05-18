//===-- Unittests for floorf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/floorf.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;
using Properties = __llvm_libc::fputil::FloatProperties<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::doublePrecision, 0,
                                           0};

TEST(FloorfTest, SpecialNumbers) {
  EXPECT_EQ(
      BitPatterns::aQuietNaN,
      valueAsBits(__llvm_libc::floorf(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::floorf(
                valueFromBits(BitPatterns::aNegativeQuietNaN))));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::floorf(
                valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(BitPatterns::aNegativeSignallingNaN,
            valueAsBits(__llvm_libc::floorf(
                valueFromBits(BitPatterns::aNegativeSignallingNaN))));

  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::floorf(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(BitPatterns::negInf, valueAsBits(__llvm_libc::floorf(
                                     valueFromBits(BitPatterns::negInf))));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::floorf(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(BitPatterns::negZero, valueAsBits(__llvm_libc::floorf(
                                      valueFromBits(BitPatterns::negZero))));
}

TEST(floorfTest, RoundedNumbers) {
  EXPECT_EQ(valueAsBits(1.0f), valueAsBits(__llvm_libc::floorf(1.0f)));
  EXPECT_EQ(valueAsBits(-1.0f), valueAsBits(__llvm_libc::floorf(-1.0f)));
  EXPECT_EQ(valueAsBits(10.0f), valueAsBits(__llvm_libc::floorf(10.0f)));
  EXPECT_EQ(valueAsBits(-10.0f), valueAsBits(__llvm_libc::floorf(-10.0f)));
  EXPECT_EQ(valueAsBits(12345.0f), valueAsBits(__llvm_libc::floorf(12345.0f)));
  EXPECT_EQ(valueAsBits(-12345.0f),
            valueAsBits(__llvm_libc::floorf(-12345.0f)));
}

TEST(floorfTest, InFloatRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 1000000;
  constexpr BitsType step = UINT32_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Floor, x, __llvm_libc::floorf(x),
                      tolerance);
  }
}
