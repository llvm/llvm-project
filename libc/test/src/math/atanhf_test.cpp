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

using LlvmLibcAtanhfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcAtanhfTest, SpecialNumbers) {
  using Sign = LIBC_NAMESPACE::fputil::Sign;
  libc_errno = 0;
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(aNaN));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::atanhf(0.0f));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::atanhf(-0.0f));
  EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::atanhf(1.0f));
  EXPECT_FP_EXCEPTION(FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::atanhf(-1.0f));
  EXPECT_FP_EXCEPTION(FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  auto bt = FPBits(1.0f);
  bt.bits += 1;

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(bt.get_val()));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  bt.set_sign(Sign::NEG);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(bt.get_val()));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(2.0f));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(-2.0f));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(inf));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  bt.set_sign(Sign::NEG);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atanhf(neg_inf));
  EXPECT_FP_EXCEPTION(FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAtanhfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  const uint32_t STEP = FPBits(1.0f).uintval() / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    ASSERT_MPFR_MATCH(mpfr::Operation::Atanh, x, LIBC_NAMESPACE::atanhf(x),
                      0.5);
    ASSERT_MPFR_MATCH(mpfr::Operation::Atanh, -x, LIBC_NAMESPACE::atanhf(-x),
                      0.5);
  }
}

// For small values, atanh(x) is x.
TEST_F(LlvmLibcAtanhfTest, SmallValues) {
  float x = FPBits(uint32_t(0x17800000)).get_val();
  float result = LIBC_NAMESPACE::atanhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Atanh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);

  x = FPBits(uint32_t(0x00400000)).get_val();
  result = LIBC_NAMESPACE::atanhf(x);
  EXPECT_MPFR_MATCH(mpfr::Operation::Atanh, x, result, 0.5);
  EXPECT_FP_EQ(x, result);
}
