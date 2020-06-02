//===-- Unittests for cosf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/math.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/math/cosf.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/CPP/Array.h"
#include "utils/FPUtil/BitPatterns.h"
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

using __llvm_libc::testing::sdcomp26094Values;

namespace mpfr = __llvm_libc::testing::mpfr;

// 12 additional bits of precision over the base precision of a |float|
// value.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 12,
                                           3 * 0x1000 / 4};

TEST(CosfTest, SpecialNumbers) {
  llvmlibc_errno = 0;

  EXPECT_TRUE(
      isQuietNaN(__llvm_libc::cosf(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::cosf(valueFromBits(BitPatterns::aNegativeQuietNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isQuietNaN(
      __llvm_libc::cosf(valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::cosf(valueFromBits(BitPatterns::aNegativeSignallingNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::one,
            valueAsBits(__llvm_libc::cosf(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::one, valueAsBits(__llvm_libc::cosf(
                                  valueFromBits(BitPatterns::negZero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  llvmlibc_errno = 0;
  EXPECT_TRUE(isQuietNaN(__llvm_libc::cosf(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(llvmlibc_errno, EDOM);

  llvmlibc_errno = 0;
  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::cosf(valueFromBits(BitPatterns::negInf))));
  EXPECT_EQ(llvmlibc_errno, EDOM);
}

TEST(CosfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), tolerance);
  }
}

// For small values, cos(x) is 1.
TEST(CosfTest, SmallValues) {
  float x = valueFromBits(0x17800000U);
  float result = __llvm_libc::cosf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result, tolerance);
  EXPECT_EQ(BitPatterns::one, valueAsBits(result));

  x = valueFromBits(0x0040000U);
  result = __llvm_libc::cosf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result, tolerance);
  EXPECT_EQ(BitPatterns::one, valueAsBits(result));
}

// SDCOMP-26094: check cosf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(CosfTest, SDCOMP_26094) {
  for (uint32_t v : sdcomp26094Values) {
    float x = valueFromBits(v);
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), tolerance);
  }
}
