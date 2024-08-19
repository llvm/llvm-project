//===-- Unittests for sin -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sin.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSinTest = LIBC_NAMESPACE::testing::FPTest<double>;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcSinTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::sin(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::sin(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::sin(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::sin(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::sin(neg_zero));
  EXPECT_FP_EQ(0x1.0p-50, LIBC_NAMESPACE::sin(0x1.0p-50));
  EXPECT_FP_EQ(min_normal, LIBC_NAMESPACE::sin(min_normal));
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::sin(min_denormal));
}
