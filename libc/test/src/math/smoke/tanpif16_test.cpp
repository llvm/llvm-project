//===-- Unittests for tanpif16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/math/tanpif16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcTanpif16Test, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tanpif16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

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

  EXPECT_FP_EQ_WITH_EXCEPTION(0x1.fe8p-15f16,
                              LIBC_NAMESPACE::tanpif16(0x1.45p-16f16),
                              FE_UNDERFLOW | FE_INEXACT);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(1.0f16, LIBC_NAMESPACE::tanpif16(2.25f16), 0);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::tanpif16(0.5f16),
                              FE_DIVBYZERO);
  EXPECT_MATH_ERRNO(ERANGE);
}
