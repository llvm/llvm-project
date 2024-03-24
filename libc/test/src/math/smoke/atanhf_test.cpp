//===-- Unittests for atanhf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/atanhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcAtanhfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAtanhfTest, SpecialNumbers) {

  LIBC_NAMESPACE::libc_errno = 0;

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

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::atanhf(1.0f), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::atanhf(-1.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  auto bt = FPBits(1.0f);
  bt.set_uintval(bt.uintval() + 1);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::atanhf(bt.get_val()),
                                  FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  bt.set_sign(Sign::NEG);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::atanhf(bt.get_val()),
                                  FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::atanhf(2.0f), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::atanhf(-2.0f), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::atanhf(inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  bt.set_sign(Sign::NEG);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::atanhf(neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}
