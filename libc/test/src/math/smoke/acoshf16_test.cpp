//===-- Unittests for acoshf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/math/acoshf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAcoshf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcAcoshf16Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acoshf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acoshf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acoshf16(zero), FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acoshf16(neg_zero));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acoshf16(neg_zero),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::acoshf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acoshf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acoshf16(neg_inf),
                              FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::acoshf16(
                         LIBC_NAMESPACE::fputil::cast<float16>(1.0)));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::acoshf16(
                         LIBC_NAMESPACE::fputil::cast<float16>(0.5)));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      aNaN,
      LIBC_NAMESPACE::acoshf16(LIBC_NAMESPACE::fputil::cast<float16>(-1.0)),
      FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      aNaN,
      LIBC_NAMESPACE::acoshf16(LIBC_NAMESPACE::fputil::cast<float16>(-2.0)),
      FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      aNaN,
      LIBC_NAMESPACE::acoshf16(LIBC_NAMESPACE::fputil::cast<float16>(-3.0)),
      FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
}
