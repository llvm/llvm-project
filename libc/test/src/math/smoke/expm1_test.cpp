//===-- Unittests for e^x - 1 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/expm1.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcExpm1Test = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcExpm1Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::expm1(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::expm1(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0, LIBC_NAMESPACE::expm1(neg_inf));
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::expm1(0x1.0p20),
                              FE_OVERFLOW);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::expm1(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::expm1(neg_zero));
  // |x| < 2^-53, expm1(x) = x
  EXPECT_FP_EQ(-0x1.23456789abcdep-55,
               LIBC_NAMESPACE::expm1(-0x1.23456789abcdep-55));
  EXPECT_FP_EQ(0x1.23456789abcdep-55,
               LIBC_NAMESPACE::expm1(0x1.23456789abcdep-55));
  // log(2^-54)
  EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::expm1(-0x1.2b708872320e2p5));
}
