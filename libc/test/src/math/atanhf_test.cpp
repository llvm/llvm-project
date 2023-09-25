//===-- Unittests for atanhf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/atanhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcAtanhfTest, SpecialNumbers) {
  libc_errno = 0;
  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(aNaN));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, __llvm_libc::atanhf(0.0f));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, __llvm_libc::atanhf(-0.0f));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(inf, __llvm_libc::atanhf(1.0f));
  EXPECT_FP_EXCEPTION(FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, __llvm_libc::atanhf(-1.0f));
  EXPECT_FP_EXCEPTION(FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  auto bt = FPBits(1.0f);
  bt.bits += 1;

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(bt.get_val()));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  bt.set_sign(true);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(bt.get_val()));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(2.0f));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(-2.0f));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(inf));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  bt.set_sign(true);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, __llvm_libc::atanhf(neg_inf));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}

TEST(LlvmLibcAtanhfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  const uint32_t STEP = FPBits(1.0f).uintval() / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    ASSERT_MPFR_MATCH(mpfr::Operation::Atanh, x, __llvm_libc::atanhf(x), 0.5);
    ASSERT_MPFR_MATCH(mpfr::Operation::Atanh, -x, __llvm_libc::atanhf(-x), 0.5);
  }
}

// For small values, atanh(x) is x.
TEST(LlvmLibcAtanhfTest, SmallValues) {
  float x = float(FPBits(uint32_t(0x17800000)));
  float result = __llvm_libc::atanhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Atanh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);

  x = float(FPBits(uint32_t(0x00400000)));
  result = __llvm_libc::atanhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Atanh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);
}
