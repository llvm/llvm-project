//===-- Unittests for sincosf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/sincosf.h"
#include "test/UnitTest/FPTest.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcSinCosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcSinCosfTest, SpecialNumbers) {
  float sin, cos;

  EXPECT_NO_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      LIBC_NAMESPACE::sincosf(aNaN, &sin, &cos));
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);

  EXPECT_NO_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      LIBC_NAMESPACE::sincosf(0.0f, &sin, &cos));
  EXPECT_FP_EQ(1.0f, cos);
  EXPECT_FP_EQ(0.0f, sin);

  EXPECT_NO_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      LIBC_NAMESPACE::sincosf(-0.0f, &sin, &cos));
  EXPECT_FP_EQ(1.0f, cos);
  EXPECT_FP_EQ(-0.0f, sin);

  EXPECT_ERRNO_FP_EXCEPT_ALL_ROUNDING(EDOM, FE_INVALID,
                                      LIBC_NAMESPACE::sincosf(inf, &sin, &cos));
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);

  EXPECT_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      EDOM, FE_INVALID, LIBC_NAMESPACE::sincosf(neg_inf, &sin, &cos));
  EXPECT_FP_EQ(aNaN, cos);
  EXPECT_FP_EQ(aNaN, sin);
}
