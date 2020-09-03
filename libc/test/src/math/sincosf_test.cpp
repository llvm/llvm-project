//===-- Unittests for sincosf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/math.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/math/sincosf.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/CPP/Array.h"
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

using __llvm_libc::testing::sdcomp26094Values;

namespace mpfr = __llvm_libc::testing::mpfr;

// 12 additional bits of precision over the base precision of a |float|
// value.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 12,
                                           3 * 0x1000 / 4};

TEST(SinCosfTest, SpecialNumbers) {
  llvmlibc_errno = 0;
  float sin, cos;

  __llvm_libc::sincosf(valueFromBits(BitPatterns::aQuietNaN), &sin, &cos);
  EXPECT_TRUE(isQuietNaN(cos));
  EXPECT_TRUE(isQuietNaN(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(valueFromBits(BitPatterns::aNegativeQuietNaN), &sin,
                       &cos);
  EXPECT_TRUE(isNegativeQuietNaN(cos));
  EXPECT_TRUE(isNegativeQuietNaN(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(valueFromBits(BitPatterns::aSignallingNaN), &sin, &cos);
  EXPECT_TRUE(isQuietNaN(cos));
  EXPECT_TRUE(isQuietNaN(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(valueFromBits(BitPatterns::aNegativeSignallingNaN), &sin,
                       &cos);
  EXPECT_TRUE(isNegativeQuietNaN(cos));
  EXPECT_TRUE(isNegativeQuietNaN(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(valueFromBits(BitPatterns::zero), &sin, &cos);
  EXPECT_EQ(BitPatterns::one, valueAsBits(cos));
  EXPECT_EQ(BitPatterns::zero, valueAsBits(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(valueFromBits(BitPatterns::negZero), &sin, &cos);
  EXPECT_EQ(BitPatterns::one, valueAsBits(cos));
  EXPECT_EQ(BitPatterns::negZero, valueAsBits(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  llvmlibc_errno = 0;
  __llvm_libc::sincosf(valueFromBits(BitPatterns::inf), &sin, &cos);
  EXPECT_TRUE(isQuietNaN(cos));
  EXPECT_TRUE(isQuietNaN(sin));
  EXPECT_EQ(llvmlibc_errno, EDOM);

  llvmlibc_errno = 0;
  __llvm_libc::sincosf(valueFromBits(BitPatterns::negInf), &sin, &cos);
  EXPECT_TRUE(isQuietNaN(cos));
  EXPECT_TRUE(isQuietNaN(sin));
  EXPECT_EQ(llvmlibc_errno, EDOM);
}

TEST(SinCosfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;

    float sin, cos;
    __llvm_libc::sincosf(x, &sin, &cos);
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, cos, tolerance);
    ASSERT_MPFR_MATCH(mpfr::Operation::Sin, x, sin, tolerance);
  }
}

// For small values, cos(x) is 1 and sin(x) is x.
TEST(SinCosfTest, SmallValues) {
  uint32_t bits = 0x17800000;
  float x = valueFromBits(bits);
  float result_cos, result_sin;
  __llvm_libc::sincosf(x, &result_sin, &result_cos);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result_cos, tolerance);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, result_sin, tolerance);
  EXPECT_EQ(BitPatterns::one, valueAsBits(result_cos));
  EXPECT_EQ(bits, valueAsBits(result_sin));

  bits = 0x00400000;
  x = valueFromBits(bits);
  __llvm_libc::sincosf(x, &result_sin, &result_cos);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result_cos, tolerance);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, result_sin, tolerance);
  EXPECT_EQ(BitPatterns::one, valueAsBits(result_cos));
  EXPECT_EQ(bits, valueAsBits(result_sin));
}

// SDCOMP-26094: check sinf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(SinCosfTest, SDCOMP_26094) {
  for (uint32_t v : sdcomp26094Values) {
    float x = valueFromBits(v);
    float sin, cos;
    __llvm_libc::sincosf(x, &sin, &cos);
    EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, cos, tolerance);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, sin, tolerance);
  }
}
