//===-- Unittests for sincosf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/math/math_utils.h"
#include "src/math/sincosf.h"
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

TEST(SinCosfTest, SpecialNumbers) {
  llvmlibc_errno = 0;
  float sin, cos;

  __llvm_libc::sincosf(as_float(FloatBits::QNan), &sin, &cos);
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(cos)));
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(sin)));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(as_float(FloatBits::NegQNan), &sin, &cos);
  EXPECT_TRUE(FloatBits::isNegQNan(as_uint32_bits(cos)));
  EXPECT_TRUE(FloatBits::isNegQNan(as_uint32_bits(sin)));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(as_float(FloatBits::SNan), &sin, &cos);
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(cos)));
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(sin)));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(as_float(FloatBits::NegSNan), &sin, &cos);
  EXPECT_TRUE(FloatBits::isNegQNan(as_uint32_bits(cos)));
  EXPECT_TRUE(FloatBits::isNegQNan(as_uint32_bits(sin)));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(as_float(FloatBits::Zero), &sin, &cos);
  EXPECT_EQ(FloatBits::One, as_uint32_bits(cos));
  EXPECT_EQ(FloatBits::Zero, as_uint32_bits(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  __llvm_libc::sincosf(as_float(FloatBits::NegZero), &sin, &cos);
  EXPECT_EQ(FloatBits::One, as_uint32_bits(cos));
  EXPECT_EQ(FloatBits::NegZero, as_uint32_bits(sin));
  EXPECT_EQ(llvmlibc_errno, 0);

  llvmlibc_errno = 0;
  __llvm_libc::sincosf(as_float(FloatBits::Inf), &sin, &cos);
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(cos)));
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(sin)));
  EXPECT_EQ(llvmlibc_errno, EDOM);

  llvmlibc_errno = 0;
  __llvm_libc::sincosf(as_float(FloatBits::NegInf), &sin, &cos);
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(cos)));
  EXPECT_TRUE(FloatBits::isQNan(as_uint32_bits(sin)));
  EXPECT_EQ(llvmlibc_errno, EDOM);
}

TEST(SinCosfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = as_float(v);
    if (isnan(x) || isinf(x))
      continue;

    float sin, cos;
    __llvm_libc::sincosf(x, &sin, &cos);
    EXPECT_TRUE(mpfr::equalsCos(x, cos, tolerance));
    EXPECT_TRUE(mpfr::equalsSin(x, sin, tolerance));
  }
}

// For small values, cos(x) is 1 and sin(x) is x.
TEST(SinCosfTest, SmallValues) {
  uint32_t bits = 0x17800000;
  float x = as_float(bits);
  float result_cos, result_sin;
  __llvm_libc::sincosf(x, &result_sin, &result_cos);
  EXPECT_TRUE(mpfr::equalsCos(x, result_cos, tolerance));
  EXPECT_TRUE(mpfr::equalsSin(x, result_sin, tolerance));
  EXPECT_EQ(FloatBits::One, as_uint32_bits(result_cos));
  EXPECT_EQ(bits, as_uint32_bits(result_sin));

  bits = 0x00400000;
  x = as_float(bits);
  __llvm_libc::sincosf(x, &result_sin, &result_cos);
  EXPECT_TRUE(mpfr::equalsCos(x, result_cos, tolerance));
  EXPECT_TRUE(mpfr::equalsSin(x, result_sin, tolerance));
  EXPECT_EQ(FloatBits::One, as_uint32_bits(result_cos));
  EXPECT_EQ(bits, as_uint32_bits(result_sin));
}

// SDCOMP-26094: check sinf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(SinCosfTest, SDCOMP_26094) {
  for (uint32_t v : sdcomp26094Values) {
    float x = as_float(v);
    float sin, cos;
    __llvm_libc::sincosf(x, &sin, &cos);
    EXPECT_TRUE(mpfr::equalsCos(x, cos, tolerance));
    EXPECT_TRUE(mpfr::equalsSin(x, sin, tolerance));
  }
}
