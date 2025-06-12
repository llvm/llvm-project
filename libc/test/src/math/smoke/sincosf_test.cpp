//===-- Unittests for sincosf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/sincosf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcSinCosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcSinCosfTest, SpecialNumbers) {
  libc_errno = 0;
  float sin, cos;

  LIBC_NAMESPACE::sincosf(sNaN, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::sincosf(aNaN, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::sincosf(0.0f, &sin, &cos);
  EXPECT_FP_EQ(1.0f, cos);
  EXPECT_FP_EQ(0.0f, sin);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::sincosf(-0.0f, &sin, &cos);
  EXPECT_FP_EQ(1.0f, cos);
  EXPECT_FP_EQ(-0.0f, sin);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::sincosf(inf, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(EDOM);

  LIBC_NAMESPACE::sincosf(neg_inf, &sin, &cos);
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
  EXPECT_MATH_ERRNO(EDOM);
}
