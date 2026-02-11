//===-- Unittests for erfcf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/math/erfcf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcErfcTest = LIBC_NAMESPACE::testing::FPTest<float16>;

TEST_F(LlvmLibcErfcTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::erfcf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::erfcf16(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f16, LIBC_NAMESPACE::erfcf16(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0f16, LIBC_NAMESPACE::erfcf16(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f16, LIBC_NAMESPACE::erfcf16(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0f16, LIBC_NAMESPACE::erfcf16(neg_zero));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcErfcTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::erfcf16(min_denormal));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::erfcf16(max_denormal));
}

TEST_F(LlvmLibcErfcTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::erfcf16(min_denormal));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::erfcf16(max_denormal));
}

TEST_F(LlvmLibcErfcTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::erfcf16(min_denormal));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::erfcf16(max_denormal));
}

#endif
