//===-- Unittests for erfcf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/erfcf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcErfcfTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcErfcfTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::erfcf(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::erfcf(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::erfcf(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0f, LIBC_NAMESPACE::erfcf(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::erfcf(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f, LIBC_NAMESPACE::erfcf(neg_zero));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcErfcfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::erfcf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::erfcf(max_denormal));
}

TEST_F(LlvmLibcErfcfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::erfcf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::erfcf(max_denormal));
}

TEST_F(LlvmLibcErfcfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::erfcf(min_denormal));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::erfcf(max_denormal));
}

#endif
