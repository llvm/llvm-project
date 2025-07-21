//===-- Unittests for shared math functions -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/math.h"
#include "test/UnitTest/FPMatcher.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

TEST(LlvmLibcSharedMathTest, AllFloat16) {
  int exponent;

  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::acoshf16(1.0f));

  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::exp10f16(0.0f16));

  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::expf16(0.0f16));

  ASSERT_FP_EQ(float16(8 << 5), LIBC_NAMESPACE::shared::ldexpf16(float(8), 5));
  ASSERT_FP_EQ(float16(-1 * (8 << 5)),
               LIBC_NAMESPACE::shared::ldexpf16(float(-8), 5));

  EXPECT_FP_EQ_ALL_ROUNDING(0.75f16,
                            LIBC_NAMESPACE::shared::frexpf16(24.0f, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_FP_EQ(0x1.921fb6p+0f16, LIBC_NAMESPACE::shared::acosf16(0.0f16));
}

#endif

TEST(LlvmLibcSharedMathTest, AllFloat) {
  int exponent;

  EXPECT_FP_EQ(0x1.921fb6p+0, LIBC_NAMESPACE::shared::acosf(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::exp10f(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::expf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::erff(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::acoshf(1.0f));

  EXPECT_FP_EQ_ALL_ROUNDING(0.75f,
                            LIBC_NAMESPACE::shared::frexpf(24.0f, &exponent));
  EXPECT_EQ(exponent, 5);

  ASSERT_FP_EQ(float(8 << 5), LIBC_NAMESPACE::shared::ldexpf(float(8), 5));
  ASSERT_FP_EQ(float(-1 * (8 << 5)),
               LIBC_NAMESPACE::shared::ldexpf(float(-8), 5));
}

TEST(LlvmLibcSharedMathTest, AllDouble) {
  EXPECT_FP_EQ(0x1.921fb54442d18p+0, LIBC_NAMESPACE::shared::acos(0.0));
  EXPECT_FP_EQ(0x1p+0, LIBC_NAMESPACE::shared::exp(0.0));
  EXPECT_FP_EQ(0x1p+0, LIBC_NAMESPACE::shared::exp10(0.0));
}

TEST(LlvmLibcSharedMathTest, AllFloat128) {
  int exponent;

  EXPECT_FP_EQ_ALL_ROUNDING(
      float128(0.75), LIBC_NAMESPACE::shared::frexpf128(24.0f, &exponent));
  EXPECT_EQ(exponent, 5);

  ASSERT_FP_EQ(float128(8 << 5),
               LIBC_NAMESPACE::shared::ldexpf128(float(8), 5));
  ASSERT_FP_EQ(float128(-1 * (8 << 5)),
               LIBC_NAMESPACE::shared::ldexpf128(float(-8), 5));
}
