//===-- Unittests for sinhf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/CPP/array.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/sinhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcSinhfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcSinhfTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinhf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::sinhf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::sinhf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::sinhf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::sinhf(neg_inf));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcSinhfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Sinh, x, LIBC_NAMESPACE::sinhf(x), 0.5);
  }
}

// For small values, sinh(x) is x.
TEST_F(LlvmLibcSinhfTest, SmallValues) {
  float x = FPBits(uint32_t(0x17800000)).get_val();
  float result = LIBC_NAMESPACE::sinhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);

  x = FPBits(uint32_t(0x00400000)).get_val();
  result = LIBC_NAMESPACE::sinhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Sinh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);
}

TEST_F(LlvmLibcSinhfTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::sinhf(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::sinhf(FPBits(0x42cffff8U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::sinhf(FPBits(0x42d00008U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcSinhfTest, ExceptionalValues) {
  float x = FPBits(uint32_t(0x3a12'85ffU)).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinh, x,
                                 LIBC_NAMESPACE::sinhf(x), 0.5);

  x = -FPBits(uint32_t(0x3a12'85ffU)).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinh, x,
                                 LIBC_NAMESPACE::sinhf(x), 0.5);
}
