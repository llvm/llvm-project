//===-- Unittests for log10 -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log10.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcLog10Test = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcLog10Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log10(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log10(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log10(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log10(0.0),
                              FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log10(-0.0),
                              FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log10(-1.0), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log10(1.0));

  double x = 1.0;
  for (int i = 0; i < 11; ++i, x *= 10.0) {
    EXPECT_FP_EQ_ALL_ROUNDING(static_cast<double>(i), LIBC_NAMESPACE::log10(x));
  }
}
