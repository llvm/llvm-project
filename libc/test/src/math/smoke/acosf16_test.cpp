//===-- Unittests for acosf16 ---------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/acosf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAcosf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAcosf16Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acosf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acosf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::acosf16(1.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acosf16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acosf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acosf16(2.0f));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acosf16(-2.0f));
  EXPECT_MATH_ERRNO(EDOM);
}
