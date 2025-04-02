//===-- Unittests for tanpif16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/tanpif16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcTanpif16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::tanpif16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::tanpif16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::tanpif16(neg_zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::tanpif16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::tanpif16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}
