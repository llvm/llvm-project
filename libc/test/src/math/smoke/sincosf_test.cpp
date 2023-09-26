//===-- Unittests for sincosf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/sincosf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = LIBC_NAMESPACE::fputil::FPBits<float>;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcSinCosfTest, SpecialNumbers) {
  libc_errno = 0;
  float sin, cos;

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
