//===-- Unittests for hypotf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "src/math/hypotf16.h"

using LlvmLibcHypotf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcHypotf16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::hypotf16(inf, aNaN));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::hypotf16(aNaN, neg_inf));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::hypotf16(aNaN, aNaN));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::hypotf16(sNaN, zero));
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::hypotf16(neg_zero, sNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::hypotf16(inf, sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::hypotf16(sNaN, neg_inf), FE_INVALID);
  EXPECT_MATH_ERRNO(0);
}
