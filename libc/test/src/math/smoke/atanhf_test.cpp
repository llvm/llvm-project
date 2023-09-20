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
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

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

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, __llvm_libc::atanhf(1.0f), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, __llvm_libc::atanhf(-1.0f),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  auto bt = FPBits(1.0f);
  bt.bits += 1;

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::atanhf(bt.get_val()),
                                  FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  bt.set_sign(true);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::atanhf(bt.get_val()),
                                  FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::atanhf(2.0f), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::atanhf(-2.0f), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::atanhf(inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  bt.set_sign(true);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(__llvm_libc::atanhf(neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}
