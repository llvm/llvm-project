//===-- Unittests for atan2f16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atan2f16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtan2f16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

static constexpr float16 one =
    LIBC_NAMESPACE::fputil::FPBits<float16>::one().get_val();
static constexpr float16 neg_one =
    LIBC_NAMESPACE::fputil::FPBits<float16>::one(LIBC_NAMESPACE::Sign::NEG)
        .get_val();

TEST_F(LlvmLibcAtan2f16Test, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atan2f16(sNaN, sNaN),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atan2f16(sNaN, one),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::atan2f16(one, sNaN),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::atan2f16(aNaN, zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::atan2f16(one, aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::atan2f16(zero, zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atan2f16(neg_zero, zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::atan2f16(one, inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(neg_zero, LIBC_NAMESPACE::atan2f16(neg_one, inf));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcAtan2f16Test, InRange) {
  // atan2(1, 1) = pi/4
  EXPECT_FP_EQ(0x1.92p-1f, LIBC_NAMESPACE::atan2f16(one, one));
  EXPECT_MATH_ERRNO(0);

  // atan2(0, 1) = 0
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::atan2f16(zero, one));
  EXPECT_MATH_ERRNO(0);
}
