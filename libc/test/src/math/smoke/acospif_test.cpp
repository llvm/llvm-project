//===-- Unittests for acospif ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/acospif.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcAcospifTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAcospifTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acospif(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acospif(aNaN));
  EXPECT_MATH_ERRNO(0);

  // acospif(0) = 0.5
  EXPECT_FP_EQ(0.5f, LIBC_NAMESPACE::acospif(0.0f));
  EXPECT_MATH_ERRNO(0);
  // acospif(-0) = 0.5
  EXPECT_FP_EQ(0.5f, LIBC_NAMESPACE::acospif(-0.0f));
  EXPECT_MATH_ERRNO(0);
  // acospif(1) = 0
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::acospif(1.0f));
  EXPECT_MATH_ERRNO(0);
  // acospif(-1) = 1
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::acospif(-1.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acospif(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acospif(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acospif(2.0f));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acospif(-2.0f));
  EXPECT_MATH_ERRNO(EDOM);
}
