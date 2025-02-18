//===-- Unittests for hypotf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypotf.h"

using LlvmLibcHypotfTest = HypotTestTemplate<float>;

TEST_F(LlvmLibcHypotfTest, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::hypotf);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcHypotfTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(min_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(min_denormal, max_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(max_denormal, min_denormal));
  EXPECT_FP_EQ(0x1.6a09e4p-126f,
               LIBC_NAMESPACE::hypotf(max_denormal, max_denormal));
}

TEST_F(LlvmLibcHypotfTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(min_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(min_denormal, max_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(max_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(max_denormal, max_denormal));
}

TEST_F(LlvmLibcHypotfTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(min_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(min_denormal, max_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(max_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::hypotf(max_denormal, max_denormal));
}

#endif
