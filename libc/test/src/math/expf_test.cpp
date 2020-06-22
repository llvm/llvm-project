//===-- Unittests for expf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/math.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/math/expf.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/ClassificationFunctions.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

#include <stdint.h>

using __llvm_libc::fputil::isNegativeQuietNaN;
using __llvm_libc::fputil::isQuietNaN;
using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

// 12 additional bits of precision over the base precision of a |float|
// value.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 12,
                                           0xFFF};

TEST(ExpfTest, SpecialNumbers) {
  llvmlibc_errno = 0;

  EXPECT_TRUE(
      isQuietNaN(__llvm_libc::expf(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::expf(valueFromBits(BitPatterns::aNegativeQuietNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isQuietNaN(
      __llvm_libc::expf(valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::expf(valueFromBits(BitPatterns::aNegativeSignallingNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expf(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::expf(valueFromBits(BitPatterns::negInf))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::one,
            valueAsBits(__llvm_libc::expf(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::one, valueAsBits(__llvm_libc::expf(
                                  valueFromBits(BitPatterns::negZero))));
  EXPECT_EQ(llvmlibc_errno, 0);
}

TEST(ExpfTest, Overflow) {
  llvmlibc_errno = 0;
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expf(valueFromBits(0x7f7fffffU))));
  EXPECT_EQ(llvmlibc_errno, ERANGE);

  llvmlibc_errno = 0;
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expf(valueFromBits(0x42cffff8U))));
  EXPECT_EQ(llvmlibc_errno, ERANGE);

  llvmlibc_errno = 0;
  EXPECT_EQ(BitPatterns::inf,
            valueAsBits(__llvm_libc::expf(valueFromBits(0x42d00008U))));
  EXPECT_EQ(llvmlibc_errno, ERANGE);
}

TEST(ExpfTest, Underflow) {
  llvmlibc_errno = 0;
  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::expf(valueFromBits(0xff7fffffU))));
  EXPECT_EQ(llvmlibc_errno, ERANGE);

  llvmlibc_errno = 0;
  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::expf(valueFromBits(0xc2cffff8U))));
  EXPECT_EQ(llvmlibc_errno, ERANGE);

  llvmlibc_errno = 0;
  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::expf(valueFromBits(0xc2d00008U))));
  EXPECT_EQ(llvmlibc_errno, ERANGE);
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST(ExpfTest, Borderline) {
  float x;

  llvmlibc_errno = 0;
  x = valueFromBits(0x42affff8U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Exp, x, __llvm_libc::expf(x), tolerance);
  EXPECT_EQ(llvmlibc_errno, 0);

  x = valueFromBits(0x42b00008U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Exp, x, __llvm_libc::expf(x), tolerance);
  EXPECT_EQ(llvmlibc_errno, 0);

  x = valueFromBits(0xc2affff8U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Exp, x, __llvm_libc::expf(x), tolerance);
  EXPECT_EQ(llvmlibc_errno, 0);

  x = valueFromBits(0xc2b00008U);
  ASSERT_MPFR_MATCH(mpfr::Operation::Exp, x, __llvm_libc::expf(x), tolerance);
  EXPECT_EQ(llvmlibc_errno, 0);
}

TEST(ExpfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;
    llvmlibc_errno = 0;
    float result = __llvm_libc::expf(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || llvmlibc_errno != 0)
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Exp, x, __llvm_libc::expf(x), tolerance);
  }
}
