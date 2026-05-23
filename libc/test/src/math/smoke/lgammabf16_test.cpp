//===-- Unittests for lgammabf16 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/lgammabf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

class LlvmLibcLgammaBf16Test : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(bfloat16)
public:
  void test_special_numbers() {
    EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::lgammabf16(aNaN));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(zero));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(neg_zero));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(inf));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(neg_inf));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::lgammabf16(bfloat16(1.0f)));
    EXPECT_FP_EQ(zero, LIBC_NAMESPACE::lgammabf16(bfloat16(2.0f)));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-1.0f)));
    EXPECT_FP_EQ(inf, LIBC_NAMESPACE::lgammabf16(bfloat16(-2.0f)));
  }
};

TEST_F(LlvmLibcLgammaBf16Test, SpecialNumbers) { test_special_numbers(); }
