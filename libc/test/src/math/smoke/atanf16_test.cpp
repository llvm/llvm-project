//===-- Unittests for atanf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/math/atanf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtanf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAtanf16Test, SpecialNumbers) {
  libc_errno = 0;
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::atanf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atanf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::atanf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atanf16(neg_zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0x1.92p0, LIBC_NAMESPACE::atanf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0x1.92p0, LIBC_NAMESPACE::atanf16(neg_inf));
  EXPECT_MATH_ERRNO(0);
}
