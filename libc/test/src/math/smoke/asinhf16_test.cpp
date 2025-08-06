//===-- Unittests for asinhf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/math/asinhf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAsinhf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAsinhf16Test, SpecialNumbers) {
  libc_errno = 0;
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::asinhf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::asinhf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::asinhf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::asinhf16(neg_zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::asinhf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::asinhf16(neg_inf));
  EXPECT_MATH_ERRNO(0);
}
