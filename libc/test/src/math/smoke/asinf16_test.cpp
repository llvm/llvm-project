//===-- Unittests for asinf16 ---------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/asinf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAsinf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAsinf16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::asinf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asinf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::asinf16(neg_zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinf16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinf16(2.0f));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinf16(-2.0f));
  EXPECT_MATH_ERRNO(EDOM);
}
