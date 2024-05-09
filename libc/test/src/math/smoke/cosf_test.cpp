//===-- Unittests for cosf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/cosf.h"
#include "test/UnitTest/FPTest.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcCosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcCosfTest, SpecialNumbers) {
  EXPECT_NO_ERRNO_FP_EXCEPT(EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cosf(aNaN)));

  EXPECT_NO_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(0.0f)));

  EXPECT_NO_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(-0.0f)));

  EXPECT_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      EDOM, FE_INVALID, EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cosf(inf)));

  EXPECT_ERRNO_FP_EXCEPT_ALL_ROUNDING(
      EDOM, FE_INVALID, EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cosf(neg_inf)));
}
