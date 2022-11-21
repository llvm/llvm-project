//===-- Unittests for pow -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/pow.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<double>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcAsinTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::pow(aNaN, aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0, __llvm_libc::pow(1.0, 1.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0, __llvm_libc::pow(1.0, 2.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, __llvm_libc::pow(2.0, inf));
}
