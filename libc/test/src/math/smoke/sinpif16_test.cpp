//===-- Unittests for sinpif16 ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------==//

#include "src/math/sinpif16.h"
#include "test/UnitTest/FPMatcher.h"
#include "src/errno/libc_errno.h"

#include <stdint.h>

using LlvmLibcSinpif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcSinpif16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::sinpif16(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::sinpif16(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNan, LIBC_NAMESPACE::sinpif16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcSinpif16Test, Integers) {
  EXPECT_FP_EQ(-0.0, LIBC_NAMESPACE::sinpif16(-0x420));
  EXPECT_FP_EQ(-0.0, LIBC_NAMESPACE::sinpif16(-0x1p+43));
  EXPECT_FP_EQ(-0.0, LIBC_NAMESPACE::sinpif16(-0x1.4p+64));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinpif16(0x420));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinpif16(0x1.cp+106));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinpif16(0x1.cp+21));
}
