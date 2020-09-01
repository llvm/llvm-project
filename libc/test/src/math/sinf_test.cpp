//===-- Unittests for sinf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/math.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/math/sinf.h"
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

TEST(SinfTest, SpecialNumbers) {
  llvmlibc_errno = 0;

  EXPECT_TRUE(
      isQuietNaN(__llvm_libc::sinf(valueFromBits(BitPatterns::aQuietNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::sinf(valueFromBits(BitPatterns::aNegativeQuietNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isQuietNaN(
      __llvm_libc::sinf(valueFromBits(BitPatterns::aSignallingNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(isNegativeQuietNaN(
      __llvm_libc::sinf(valueFromBits(BitPatterns::aNegativeSignallingNaN))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::zero,
            valueAsBits(__llvm_libc::sinf(valueFromBits(BitPatterns::zero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(BitPatterns::negZero, valueAsBits(__llvm_libc::sinf(
                                      valueFromBits(BitPatterns::negZero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  llvmlibc_errno = 0;
  EXPECT_TRUE(isQuietNaN(__llvm_libc::sinf(valueFromBits(BitPatterns::inf))));
  EXPECT_EQ(llvmlibc_errno, EDOM);

  llvmlibc_errno = 0;
  EXPECT_TRUE(
      isQuietNaN(__llvm_libc::sinf(valueFromBits(BitPatterns::negInf))));
  EXPECT_EQ(llvmlibc_errno, EDOM);
}

TEST(SinfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Sin, x, __llvm_libc::sinf(x), 1.0);
  }
}

TEST(SinfTest, SpecificBitPatterns) {
  float x = valueFromBits(0xc70d39a1);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, __llvm_libc::sinf(x), 1.0);
}

// For small values, sin(x) is x.
TEST(SinfTest, SmallValues) {
  uint32_t bits = 0x17800000;
  float x = valueFromBits(bits);
  float result = __llvm_libc::sinf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, result, 1.0);
  EXPECT_EQ(bits, valueAsBits(result));

  bits = 0x00400000;
  x = valueFromBits(bits);
  result = __llvm_libc::sinf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, result, 1.0);
  EXPECT_EQ(bits, valueAsBits(result));
}

// SDCOMP-26094: check sinf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(SinfTest, SDCOMP_26094) {
  for (uint32_t v : sdcomp26094Values) {
    float x = valueFromBits(v);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sin, x, __llvm_libc::sinf(x), 1.0);
  }
}
