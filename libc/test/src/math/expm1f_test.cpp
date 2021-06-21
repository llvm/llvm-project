//===-- Unittests for expm1f-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/expm1f.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/ClassificationFunctions.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using __llvm_libc::fputil::isNegativeQuietNaN;
using __llvm_libc::fputil::isQuietNaN;
using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

TEST(LlvmLibcExpm1fTest, SpecialNumbers) {
  errno = 0;

  EXPECT_TRUE(
      isQuietNaN(__llvm_libc::expm1f(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::expm1f(valueFromBits(BitPatterns::aNegativeQuietNaN))));
  EXPECT_EQ(errno, 0);

  EXPECT_TRUE(isQuietNaN(
      __llvm_libc::expm1f(valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::expm1f(valueFromBits(BitPatterns::aNegativeSignallingNaN))));
  EXPECT_EQ(errno, 0);

  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(errno, 0);

  EXPECT_EQ(BitPatterns::negOne, valueAsBits(__llvm_libc::expm1f(
                                     valueFromBits(BitPatterns::negInf))));
  EXPECT_EQ(errno, 0);

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(errno, 0);

  EXPECT_EQ(BitPatterns::negZero, valueAsBits(__llvm_libc::expm1f(
                                      valueFromBits(BitPatterns::negZero))));
  EXPECT_EQ(errno, 0);
}

TEST(LlvmLibcExpm1fTest, Overflow) {
  errno = 0;
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(0x7f7fffffU))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(0x42cffff8U))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(0x42d00008U))));
  EXPECT_EQ(errno, ERANGE);
}

TEST(LlvmLibcExpm1fTest, Underflow) {
  errno = 0;
  EXPECT_EQ(BitPatterns::negOne,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(0xff7fffffU))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  EXPECT_EQ(BitPatterns::negOne,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(0xc2cffff8U))));
  EXPECT_EQ(errno, ERANGE);

  errno = 0;
  EXPECT_EQ(BitPatterns::negOne,
            valueAsBits(__llvm_libc::expm1f(valueFromBits(0xc2d00008U))));
  EXPECT_EQ(errno, ERANGE);
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST(LlvmLibcExpm1fTest, Borderline) {
  float x;

  errno = 0;
  x = valueFromBits(0x42affff8U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);

  x = valueFromBits(0x42b00008U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);

  x = valueFromBits(0xc2affff8U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);

  x = valueFromBits(0xc2b00008U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.0);
  EXPECT_EQ(errno, 0);
}

TEST(LlvmLibcExpm1fTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;
    errno = 0;
    float result = __llvm_libc::expm1f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || errno != 0)
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Expm1, x, __llvm_libc::expm1f(x), 1.5);
  }
}
