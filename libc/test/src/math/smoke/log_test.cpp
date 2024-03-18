//===-- Unittests for log -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcLogTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcLogTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log(0.0), FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log(-0.0), FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log(-1.0), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log(1.0));
}
