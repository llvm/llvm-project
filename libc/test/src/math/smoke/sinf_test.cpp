//===-- Unittests for sinf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/sinf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using LlvmLibcSinfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcSinfTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::sinf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::sinf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinf(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}
