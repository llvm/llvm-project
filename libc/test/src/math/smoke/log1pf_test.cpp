//===-- Unittests for log1pf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log1pf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using LlvmLibcLog1pfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcLog1pfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log1pf(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log1pf(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log1pf(neg_inf), FE_INVALID);
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::log1pf(0.0f));
  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::log1pf(-0.0f));
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log1pf(-1.0f),
                              FE_DIVBYZERO);
}
