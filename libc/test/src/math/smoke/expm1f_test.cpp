//===-- Unittests for expm1f-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/expm1f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

using LlvmLibcExpm1fTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcExpm1fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::expm1f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::expm1f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(-1.0f, LIBC_NAMESPACE::expm1f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::expm1f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::expm1f(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpm1fTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expm1f(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expm1f(FPBits(0x42cffff8U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expm1f(FPBits(0x42d00008U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcExpm1fTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(max_denormal));
}

TEST_F(LlvmLibcExpm1fTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(max_denormal));
}

TEST_F(LlvmLibcExpm1fTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::expm1f(max_denormal));
}

#endif
