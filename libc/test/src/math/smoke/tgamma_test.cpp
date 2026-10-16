//===-- Unittests for tgamma ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/tgamma.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTgammaTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcTgammaTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgamma(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgamma(neg_inf),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tgamma(-1.0), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::tgamma(zero), FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::tgamma(neg_zero),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);

  // Source: https://members.loria.fr/PZimmermann/papers/gamma.pdf
  // tgamma(0x1.573fae561f648p+7) > DBL_MAX
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::tgamma(0x1.573fae561f648p+7),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  // One ULP below 0x1.573fae561f648p+7
  // It is the largest input our function returns a finite output for
  EXPECT_FP_NE(inf, LIBC_NAMESPACE::tgamma(0x1.573fae561f647p+7));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::tgamma(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::tgamma(inf));
}
