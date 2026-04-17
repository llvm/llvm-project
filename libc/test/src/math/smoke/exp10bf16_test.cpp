//===-- Unittests for exp10bf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/exp10bf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcExp10Bf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExp10Bf16Test, SpecialNumbers) {
  EXPECT_FP_IS_NAN(LIBC_NAMESPACE::exp10bf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::exp10bf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::exp10bf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp10bf16(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(1.0), LIBC_NAMESPACE::exp10bf16(zero));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(bfloat16(1.0), LIBC_NAMESPACE::exp10bf16(neg_zero));
  EXPECT_MATH_ERRNO(0);
}

