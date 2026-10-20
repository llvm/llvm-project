//===-- Unittests for tanhf -----------------------------------------------===//
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
#include "src/math/tanhf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTanhfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcTanhfTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::tanhf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

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

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcTanhfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(max_denormal));
}

TEST_F(LlvmLibcTanhfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(max_denormal));
}

TEST_F(LlvmLibcTanhfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::tanhf(max_denormal));
}

#endif
