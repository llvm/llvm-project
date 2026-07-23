//===-- Unittests for expf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcExpfTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::expf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::expf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::expf(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::expf(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::expf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::expf(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExpfTest, Overflow) {
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expf(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expf(FPBits(0x42cffff8U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::expf(FPBits(0x42d00008U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcExpfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(max_denormal));
}

TEST_F(LlvmLibcExpfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(max_denormal));
}

TEST_F(LlvmLibcExpfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::expf(max_denormal));
}

#endif
