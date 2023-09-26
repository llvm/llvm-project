//===-- Unittests for tanhf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/tanhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = LIBC_NAMESPACE::fputil::FPBits<float>;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcTanhfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::tanhf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::tanhf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::tanhf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::tanhf(neg_inf));
  EXPECT_MATH_ERRNO(0);
}
