//===-- Unittests for e^x - 1 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/expm1.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;
using __llvm_libc::testing::tlog;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcExpm1Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::expm1(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::expm1(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0, __llvm_libc::expm1(neg_inf));
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, __llvm_libc::expm1(0x1.0p20), FE_OVERFLOW);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, __llvm_libc::expm1(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, __llvm_libc::expm1(neg_zero));
  // |x| < 2^-53, expm1(x) = x
  EXPECT_FP_EQ(-0x1.23456789abcdep-55,
               __llvm_libc::expm1(-0x1.23456789abcdep-55));
  EXPECT_FP_EQ(0x1.23456789abcdep-55,
               __llvm_libc::expm1(0x1.23456789abcdep-55));
  // log(2^-54)
  EXPECT_FP_EQ(0x1.23456789a, __llvm_libc::expm1(-0x1.2b708872320e2p5));
}
