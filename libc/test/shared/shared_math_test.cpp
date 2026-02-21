//===-- Unittests for shared math functions -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/math.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

TEST(LlvmLibcSharedMathTest, AllFloat16) {
  int exponent;

  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::acoshf16(1.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::acospif16(1.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::rsqrtf16(1.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::sqrtf16(1.0f16));

  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::asinf16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::asinhf16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::atanf16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::atanhf16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::cosf16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::coshf16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::cospif16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::exp10f16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::exp10m1f16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::exp2f16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::exp2m1f16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::expf16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::expm1f16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::hypotf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::logf16(1.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::sinhf16(0.0f16));

  EXPECT_FP_EQ(float16(10.0), LIBC_NAMESPACE::shared::f16fma(2.0, 3.0, 4.0));

  EXPECT_FP_EQ(float16(10.0),
               LIBC_NAMESPACE::shared::f16fmaf(2.0f, 3.0f, 4.0f));

#ifdef LIBC_TYPES_HAS_FLOAT128

  EXPECT_FP_EQ(10.0f16, LIBC_NAMESPACE::shared::f16fmaf128(
                            float128(2.0), float128(3.0), float128(4.0)));

  EXPECT_FP_EQ(
      5.0f16, LIBC_NAMESPACE::shared::f16addf128(float128(2.0), float128(3.0)));

#endif

  EXPECT_FP_EQ(5.0f16, LIBC_NAMESPACE::shared::f16add(2.0, 3.0));
  EXPECT_FP_EQ(5.0f16, LIBC_NAMESPACE::shared::f16addf(2.0f, 3.0f));
  EXPECT_FP_EQ(5.0f16, LIBC_NAMESPACE::shared::f16addl(2.0L, 3.0L));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::f16sqrt(0.0));

  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::f16sqrtf(0.0f));

  EXPECT_FP_EQ(float16(10.0),
               LIBC_NAMESPACE::shared::f16fmal(2.0L, 3.0L, 4.0L));

  ASSERT_FP_EQ(float16(8 << 5), LIBC_NAMESPACE::shared::ldexpf16(8.0f16, 5));
  ASSERT_FP_EQ(float16(-1 * (8 << 5)),
               LIBC_NAMESPACE::shared::ldexpf16(-8.0f16, 5));

  EXPECT_FP_EQ_ALL_ROUNDING(
      0.75f16, LIBC_NAMESPACE::shared::frexpf16(24.0f16, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbf16(1.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::log10f16(10.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::log2f16(2.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::logbf16(1.0f16));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbf16(1.0f16));

  EXPECT_FP_EQ(0x1.921fb6p+0f16, LIBC_NAMESPACE::shared::acosf16(0.0f16));
  EXPECT_FP_EQ(0x1p+0f16, LIBC_NAMESPACE::shared::f16sqrtl(1.0L));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::sinf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::tanf16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::sinpif16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::tanhf16(0.0f16));

  float16 canonicalizef16_cx = 0.0f16;
  float16 canonicalizef16_x = 0.0f16;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizef16(&canonicalizef16_cx,
                                                       &canonicalizef16_x));
  EXPECT_FP_EQ(0x0p+0f16, canonicalizef16_cx);

  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::ceilf16(0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::fmaxf16(0.0f16, 0.0f16));
}

#endif // LIBC_TYPES_HAS_FLOAT16

TEST(LlvmLibcSharedMathTest, AllFloat) {
  int exponent;
  float sin, cos;

  EXPECT_FP_EQ(0x1.921fb6p+0, LIBC_NAMESPACE::shared::acosf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::acoshf(1.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::asinf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::asinhf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::asinpif(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::atan2f(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::atanf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::atanhf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::cbrtf(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::cosf(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::coshf(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::cospif(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::exp10m1f(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::erff(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::exp10f(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::exp2m1f(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::expf(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::exp2f(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::expm1f(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::hypotf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::logf(1.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::sinhf(0.0f));

  EXPECT_FP_EQ_ALL_ROUNDING(0.75f,
                            LIBC_NAMESPACE::shared::frexpf(24.0f, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbf(1.0f));

  ASSERT_FP_EQ(float(8 << 5), LIBC_NAMESPACE::shared::ldexpf(8.0f, 5));
  ASSERT_FP_EQ(float(-1 * (8 << 5)), LIBC_NAMESPACE::shared::ldexpf(-8.0f, 5));

  EXPECT_EQ(long(0), LIBC_NAMESPACE::shared::llogbf(1.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::log1pf(0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::log10f(10.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::log2f(2.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::logbf(1.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::powf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::rsqrtf(1.0f));

  LIBC_NAMESPACE::shared::sincosf(0.0f, &sin, &cos);
  ASSERT_FP_EQ(1.0f, cos);
  ASSERT_FP_EQ(0.0f, sin);
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::sinpif(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::sinf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::sqrtf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::tanf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::tanhf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::tanpif(0.0f));

  float canonicalizef_cx = 0.0f;
  float canonicalizef_x = 0.0f;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizef(&canonicalizef_cx,
                                                     &canonicalizef_x));
  EXPECT_FP_EQ(0x0p+0f, canonicalizef_cx);

  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::ceilf(0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fmaxf(0.0f, 0.0f));
}

TEST(LlvmLibcSharedMathTest, AllDouble) {
  double sin, cos;
  LIBC_NAMESPACE::shared::sincos(0.0, &sin, &cos);
  EXPECT_FP_EQ(0x1.921fb54442d18p+0, LIBC_NAMESPACE::shared::acos(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::asin(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::atan(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::atan2(0.0, 0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::cbrt(0.0));
  EXPECT_FP_EQ(0x1p+0, LIBC_NAMESPACE::shared::cos(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::dsqrtl(0.0));
  EXPECT_FP_EQ(0x1p+0, LIBC_NAMESPACE::shared::exp(0.0));
  EXPECT_FP_EQ(0x1p+0, LIBC_NAMESPACE::shared::exp2(0.0));
  EXPECT_FP_EQ(0x1p+0, LIBC_NAMESPACE::shared::exp10(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::expm1(0.0));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::ffma(0.0, 0.0, 0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::fsqrt(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::log(1.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::log10(1.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::log1p(0.0));
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::log2(1.0));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::pow(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::sin(0.0));
  EXPECT_FP_EQ(1.0, cos);
  EXPECT_FP_EQ(0.0, sin);
  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::sqrt(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::tan(0.0));
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogb(1.0));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogb(1.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::logb(1.0));

  double canonicalize_cx = 0.0;
  double canonicalize_x = 0.0;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalize(&canonicalize_cx,
                                                    &canonicalize_x));
  EXPECT_FP_EQ(0.0, canonicalize_cx);

  EXPECT_FP_EQ(0x0p+0, LIBC_NAMESPACE::shared::ceil(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fadd(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmax(0.0, 0.0));
}

TEST(LlvmLibcSharedMathTest, AllLongDouble) {
  EXPECT_FP_EQ(0x0p+0L,
               LIBC_NAMESPACE::shared::dfmal(0x0.p+0L, 0x0.p+0L, 0x0.p+0L));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fsqrtl(0.0L));
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbl(0x1.p+0L));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbl(1.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::logbl(1.0L));
  EXPECT_FP_EQ(10.0f, LIBC_NAMESPACE::shared::ffmal(2.0L, 3.0, 4.0L));

  long double canonicalizel_cx = 0.0L;
  long double canonicalizel_x = 0.0L;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizel(&canonicalizel_cx,
                                                     &canonicalizel_x));
  EXPECT_FP_EQ(0x0p+0L, canonicalizel_cx);

  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::ceill(0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::faddl(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fmaxl(0.0L, 0.0L));
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedMathTest, AllFloat128) {
  int exponent;

  EXPECT_FP_EQ(float128(0x0p+0),
               LIBC_NAMESPACE::shared::atan2f128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(0x1p+0f, LIBC_NAMESPACE::shared::fsqrtf128(float128(1.0f)));
  EXPECT_FP_EQ_ALL_ROUNDING(float128(0.75), LIBC_NAMESPACE::shared::frexpf128(
                                                float128(24), &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(3, LIBC_NAMESPACE::shared::ilogbf128(float128(8.0)));
  ASSERT_FP_EQ(float128(8 << 5),
               LIBC_NAMESPACE::shared::ldexpf128(float128(8), 5));
  ASSERT_FP_EQ(float128(-1 * (8 << 5)),
               LIBC_NAMESPACE::shared::ldexpf128(float128(-8), 5));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::logbf128(float128(1.0)));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::dfmaf128(
                        float128(0.0), float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0x1p+0),
               LIBC_NAMESPACE::shared::sqrtf128(float128(1.0)));

  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbf128(float128(1.0)));

  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16addf128(
                                  float128(2.0), float128(3.0)));

  float128 canonicalizef128_cx = float128(0.0);
  float128 canonicalizef128_x = float128(0.0);
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizef128(&canonicalizef128_cx,
                                                        &canonicalizef128_x));
  EXPECT_FP_EQ(float128(0.0), canonicalizef128_cx);
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::ceilf128(float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::faddf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::fmaxf128(float128(0.0), float128(0.0)));
}

#endif // LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedMathTest, AllBFloat16) {
  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16add(2.0, 3.0));
  EXPECT_FP_EQ(bfloat16(2.0f), LIBC_NAMESPACE::shared::bf16divf(4.0f, 2.0f));
  EXPECT_FP_EQ(bfloat16(2.0), LIBC_NAMESPACE::shared::bf16divl(6.0L, 3.0L));
  EXPECT_FP_EQ(bfloat16(10.0),
               LIBC_NAMESPACE::shared::bf16fmal(2.0L, 3.0L, 4.0L));

  bfloat16 canonicalizebf16_cx = bfloat16(0.0);
  bfloat16 canonicalizebf16_x = bfloat16(0.0);
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizebf16(&canonicalizebf16_cx,
                                                        &canonicalizebf16_x));
  EXPECT_FP_EQ(bfloat16(0.0), canonicalizebf16_cx);
  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16addf(2.0f, 3.0f));
  EXPECT_FP_EQ(bfloat16(10.0),
               LIBC_NAMESPACE::shared::bf16fmaf(2.0f, 3.0f, 4.0f));

  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::ceilbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::fmaxbf16(bfloat16(0.0), bfloat16(0.0)));
}
