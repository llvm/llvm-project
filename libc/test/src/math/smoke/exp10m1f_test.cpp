//===-- Unittests for exp10m1f --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/exp10m1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExp10m1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcExp10m1fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_EQ(FPBits(aNaN).uintval(),
            FPBits(LIBC_NAMESPACE::exp10m1f(aNaN)).uintval());
  EXPECT_EQ(FPBits(neg_aNaN).uintval(),
            FPBits(LIBC_NAMESPACE::exp10m1f(neg_aNaN)).uintval());
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::exp10m1f(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::exp10m1f(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp10m1f(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::exp10m1f(neg_zero));

  EXPECT_FP_EQ_ALL_ROUNDING(9.0f, LIBC_NAMESPACE::exp10m1f(1.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(99.0f, LIBC_NAMESPACE::exp10m1f(2.0f));
  EXPECT_FP_EQ_ALL_ROUNDING(999.0f, LIBC_NAMESPACE::exp10m1f(3.0f));
}

TEST_F(LlvmLibcExp10m1fTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp10m1f(0x1.fffffep+127f),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp10m1f(0x1.344136p+5),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp10m1f(0x1.344138p+5),
                              FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExp10m1fTest, Underflow) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(-1.0f, LIBC_NAMESPACE::exp10m1f(-max_normal),
                              FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(-1.0f, LIBC_NAMESPACE::exp10m1f(-0x1.e1a5e4p+2f),
                              FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}
