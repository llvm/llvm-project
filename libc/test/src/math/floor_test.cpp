//===-- Unittests for floor -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/floor.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<double>;
using Properties = __llvm_libc::fputil::FloatProperties<double>;

namespace mpfr = __llvm_libc::testing::mpfr;

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::doublePrecision, 0,
                                           0};

TEST(FloorTest, SpecialNumbers) {
  EXPECT_EQ(
      BitPatterns::aQuietNaN,
      valueAsBits(__llvm_libc::floor(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::floor(
                valueFromBits(BitPatterns::aNegativeQuietNaN))));

  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::floor(
                valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(BitPatterns::aNegativeSignallingNaN,
            valueAsBits(__llvm_libc::floor(
                valueFromBits(BitPatterns::aNegativeSignallingNaN))));

  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::floor(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(BitPatterns::negInf, valueAsBits(__llvm_libc::floor(
                                     valueFromBits(BitPatterns::negInf))));

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::floor(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(BitPatterns::negZero, valueAsBits(__llvm_libc::floor(
                                      valueFromBits(BitPatterns::negZero))));
}

TEST(floorTest, RoundedNumbers) {
  EXPECT_EQ(valueAsBits(1.0), valueAsBits(__llvm_libc::floor(1.0)));
  EXPECT_EQ(valueAsBits(-1.0), valueAsBits(__llvm_libc::floor(-1.0)));
  EXPECT_EQ(valueAsBits(10.0), valueAsBits(__llvm_libc::floor(10.0)));
  EXPECT_EQ(valueAsBits(-10.0), valueAsBits(__llvm_libc::floor(-10.0)));
  EXPECT_EQ(valueAsBits(12345.0), valueAsBits(__llvm_libc::floor(12345.0)));
  EXPECT_EQ(valueAsBits(-12345.0), valueAsBits(__llvm_libc::floor(-12345.0)));
}

TEST(floorTest, InDoubleRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 1000000;
  constexpr BitsType step = UINT64_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Floor, x, __llvm_libc::floor(x),
                      tolerance);
  }
}
