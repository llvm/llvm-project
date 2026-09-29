//===-- Unittests for rsqrtf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/math/rsqrtf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcRsqrtf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
using LIBC_NAMESPACE::fputil::cast;

TEST_F(LlvmLibcRsqrtf16Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::rsqrtf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::rsqrtf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::rsqrtf16(zero));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(neg_inf, LIBC_NAMESPACE::rsqrtf16(neg_zero));
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ(
      LIBC_NAMESPACE::fputil::cast<float16>(1.0f),
      LIBC_NAMESPACE::rsqrtf16(LIBC_NAMESPACE::fputil::cast<float16>(1.0f)));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::rsqrtf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::rsqrtf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::rsqrtf16(
                         LIBC_NAMESPACE::fputil::cast<float16>(-2.0f)));
  EXPECT_MATH_ERRNO(EDOM);
}
