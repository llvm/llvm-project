//===-- Unittests for cosf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cosf.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/CPP/Array.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using __llvm_libc::testing::sdcomp26094Values;
using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcCosfTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::cosf(aNaN));
  EXPECT_EQ(errno, 0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::cosf(0.0f));
  EXPECT_EQ(errno, 0);

  EXPECT_FP_EQ(1.0f, __llvm_libc::cosf(-0.0f));
  EXPECT_EQ(errno, 0);

  errno = 0;
  EXPECT_FP_EQ(aNaN, __llvm_libc::cosf(inf));
  EXPECT_EQ(errno, EDOM);

  errno = 0;
  EXPECT_FP_EQ(aNaN, __llvm_libc::cosf(negInf));
  EXPECT_EQ(errno, EDOM);
}

TEST(LlvmLibcCosfTest, InFloatRange) {
  constexpr uint32_t count = 1000000;
  constexpr uint32_t step = UINT32_MAX / count;
  for (uint32_t i = 0, v = 0; i <= count; ++i, v += step) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), 1.0);
  }
}

// For small values, cos(x) is 1.
TEST(LlvmLibcCosfTest, SmallValues) {
  float x = float(FPBits(0x17800000U));
  float result = __llvm_libc::cosf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result, 1.0);
  EXPECT_FP_EQ(1.0f, result);

  x = float(FPBits(0x0040000U));
  result = __llvm_libc::cosf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Cos, x, result, 1.0);
  EXPECT_FP_EQ(1.0f, result);
}

// SDCOMP-26094: check cosf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST(LlvmLibcCosfTest, SDCOMP_26094) {
  for (uint32_t v : sdcomp26094Values) {
    float x = float(FPBits(v));
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, __llvm_libc::cosf(x), 1.0);
  }
}
