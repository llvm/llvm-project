//===-- Unittests for asin ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asin.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using FPBits = __llvm_libc::fputil::FPBits<double>;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcAsinTest, SpecialNumbers) {
  errno = 0;

  EXPECT_FP_EQ(aNaN, __llvm_libc::asin(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0, __llvm_libc::asin(0.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0, __llvm_libc::asin(-0.0));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, __llvm_libc::asin(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, __llvm_libc::asin(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}
