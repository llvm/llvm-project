//===-- Unittests for sinpif ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/sinpif.h"
#include "test/UnitTest/FPMatcher.h"

#include <stdint.h>

using LlvmLibcSinpifTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcSinpifTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::sinpif(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::sinpif(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcSinpifTest, Integers) {
  EXPECT_FP_EQ(-0.0, LIBC_NAMESPACE::sinpif(-0x420));
  EXPECT_FP_EQ(-0.0, LIBC_NAMESPACE::sinpif(-0x1p+43));
  EXPECT_FP_EQ(-0.0, LIBC_NAMESPACE::sinpif(-0x1.4p+64));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinpif(0x420));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinpif(0x1.cp+106));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::sinpif(0x1.cp+21));
}
