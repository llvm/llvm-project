//===-- Unittests for rsqrtf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/math/rsqrtf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcRsqrtfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcRsqrtfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::rsqrtf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::rsqrtf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::rsqrtf(zero));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::rsqrtf(neg_zero));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::rsqrtf(1.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::rsqrtf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::rsqrtf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::rsqrtf(-2.0f));
  EXPECT_MATH_ERRNO(EDOM);
}
