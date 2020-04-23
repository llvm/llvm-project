//===-- Unittests for sinf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/math/math_utils.h"
#include "src/math/sinf.h"
#include "test/src/math/float.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/CPP/Array.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

#include <stdint.h>

using __llvm_libc::as_float;
using __llvm_libc::as_uint32_bits;

using __llvm_libc::testing::FloatBits;
using __llvm_libc::testing::sdcomp26094Values;

namespace mpfr = __llvm_libc::testing::mpfr;

// 12 additional bits of precision over the base precision of a |float|
// value.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 12,
                                           3 * 0x1000 / 4};

TEST(SinfTest, SpecialNumbers) {
  llvmlibc_errno = 0;

  EXPECT_TRUE(FloatBits::isQNan(
      as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::QNan)))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(FloatBits::isNegQNan(
      as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::NegQNan)))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(FloatBits::isQNan(
      as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::SNan)))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_TRUE(FloatBits::isNegQNan(
      as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::NegSNan)))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(FloatBits::Zero,
            as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::Zero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  EXPECT_EQ(FloatBits::NegZero,
            as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::NegZero))));
  EXPECT_EQ(llvmlibc_errno, 0);

  llvmlibc_errno = 0;
  EXPECT_TRUE(FloatBits::isQNan(
      as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::Inf)))));
  EXPECT_EQ(llvmlibc_errno, EDOM);

  llvmlibc_errno = 0;
  EXPECT_TRUE(FloatBits::isNegQNan(
      as_uint32_bits(__llvm_libc::sinf(as_float(FloatBits::NegInf)))));
  EXPECT_EQ(llvmlibc_errno, EDOM);
}

TEST(SinfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = as_float(v);
    if (isnan(x) || isinf(x))
      continue;
    EXPECT_TRUE(mpfr::equalsSin(x, __llvm_libc::sinf(x), tolerance));
  }
}

TEST(SinfTest, SpecificBitPatterns) {
  float x = as_float(0xc70d39a1);
  EXPECT_TRUE(mpfr::equalsSin(x, __llvm_libc::sinf(x), tolerance));
}

// For small values, sin(x) is x.
TEST(SinfTest, SmallValues) {
  uint32_t bits = 0x17800000;
  float x = as_float(bits);
  float result = __llvm_libc::sinf(x);
  EXPECT_TRUE(mpfr::equalsSin(x, result, tolerance));
  EXPECT_EQ(bits, as_uint32_bits(result));

  bits = 0x00400000;
  x = as_float(bits);
  result = __llvm_libc::sinf(x);
  EXPECT_TRUE(mpfr::equalsSin(x, result, tolerance));
  EXPECT_EQ(bits, as_uint32_bits(result));
}

// SDCOMP-26094: check sinf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(SinfTest, SDCOMP_26094) {
  for (uint32_t v : sdcomp26094Values) {
    float x = as_float(v);
    EXPECT_TRUE(mpfr::equalsSin(x, __llvm_libc::sinf(x), tolerance));
  }
}
