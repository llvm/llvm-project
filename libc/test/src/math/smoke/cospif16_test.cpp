//===-- Unittests for cospif16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/cast.h"
#include "src/__support/libc_errno.h"
#include "src/math/cospif16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCospif16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcCospif16Test, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::cospif16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospif16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospif16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(FPBits::one().get_val(), LIBC_NAMESPACE::cospif16(neg_zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospif16(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cospif16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcCospif16Test, Integers) {
  EXPECT_FP_EQ(FPBits::one().get_val(),
               LIBC_NAMESPACE::cospif16(
                   LIBC_NAMESPACE::fputil::cast<float16>(-0x420.0p0)));
  EXPECT_FP_EQ(FPBits::one().get_val(),
               LIBC_NAMESPACE::cospif16(
                   LIBC_NAMESPACE::fputil::cast<float16>(-0x1.4p+14)));
  EXPECT_FP_EQ(FPBits::one(Sign::NEG).get_val(),
               LIBC_NAMESPACE::cospif16(
                   LIBC_NAMESPACE::fputil::cast<float16>(0x421.0p0)));
  EXPECT_FP_EQ(FPBits::one(Sign::NEG).get_val(),
               LIBC_NAMESPACE::cospif16(
                   LIBC_NAMESPACE::fputil::cast<float16>(0x333.0p0)));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::cospif16(
                         LIBC_NAMESPACE::fputil::cast<float16>(-0x1.28p4)));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::cospif16(
                         LIBC_NAMESPACE::fputil::cast<float16>(-0x1.ffcp9)));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::cospif16(
                         LIBC_NAMESPACE::fputil::cast<float16>(0x1.01p7)));
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::cospif16(
                         LIBC_NAMESPACE::fputil::cast<float16>(0x1.f6cp9)));
}
