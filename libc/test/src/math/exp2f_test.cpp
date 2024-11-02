//===-- Unittests for exp2f -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA
#include "src/errno/libc_errno.h"
#include "src/math/exp2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <stdint.h>

using LlvmLibcExp2fTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcExp2fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::exp2f(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::exp2f(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::exp2f(neg_inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::exp2f(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::exp2f(-0.0f));
  EXPECT_MATH_ERRNO(0);
}

TEST_F(LlvmLibcExp2fTest, Overflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp2f(FPBits(0x7f7fffffU).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp2f(FPBits(0x43000000U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  EXPECT_FP_EQ_WITH_EXCEPTION(
      inf, LIBC_NAMESPACE::exp2f(FPBits(0x43000001U).get_val()), FE_OVERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExp2fTest, TrickyInputs) {
  constexpr int N = 12;
  constexpr uint32_t INPUTS[N] = {
      0x3b429d37U, /*0x1.853a6ep-9f*/
      0x3c02a9adU, /*0x1.05535ap-7f*/
      0x3ca66e26U, /*0x1.4cdc4cp-6f*/
      0x3d92a282U, /*0x1.254504p-4f*/
      0x42fa0001U, /*0x1.f40002p+6f*/
      0x42ffffffU, /*0x1.fffffep+6f*/
      0xb8d3d026U, /*-0x1.a7a04cp-14f*/
      0xbcf3a937U, /*-0x1.e7526ep-6f*/
      0xc2fa0001U, /*-0x1.f40002p+6f*/
      0xc2fc0000U, /*-0x1.f8p+6f*/
      0xc2fc0001U, /*-0x1.f80002p+6f*/
      0xc3150000U, /*-0x1.2ap+7f*/
  };
  for (int i = 0; i < N; ++i) {
    LIBC_NAMESPACE::libc_errno = 0;
    float x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                   LIBC_NAMESPACE::exp2f(x), 0.5);
    EXPECT_MATH_ERRNO(0);
  }
}

TEST_F(LlvmLibcExp2fTest, Underflow) {
  LIBC_NAMESPACE::libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION(
      0.0f, LIBC_NAMESPACE::exp2f(FPBits(0xff7fffffU).get_val()), FE_UNDERFLOW);
  EXPECT_MATH_ERRNO(ERANGE);

  float x = FPBits(0xc3158000U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                 LIBC_NAMESPACE::exp2f(x), 0.5);
  EXPECT_MATH_ERRNO(0);

  x = FPBits(0xc3160000U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                 LIBC_NAMESPACE::exp2f(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);

  x = FPBits(0xc3165432U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                 LIBC_NAMESPACE::exp2f(x), 0.5);
  EXPECT_MATH_ERRNO(ERANGE);
}

TEST_F(LlvmLibcExp2fTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (isnan(x) || isinf(x))
      continue;
    LIBC_NAMESPACE::libc_errno = 0;
    float result = LIBC_NAMESPACE::exp2f(x);

    // If the computation resulted in an error or did not produce valid result
    // in the single-precision floating point range, then ignore comparing with
    // MPFR result as MPFR can still produce valid results because of its
    // wider precision.
    if (isnan(result) || isinf(result) || LIBC_NAMESPACE::libc_errno != 0)
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp2, x,
                                   LIBC_NAMESPACE::exp2f(x), 0.5);
  }
}
