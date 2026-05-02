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
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

  int exponent;
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::acoshf16(1.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::acospif16(1.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::rsqrtf16(1.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::sqrtf16(1.0f16));

  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::asinf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::asinhf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::asinpif16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::atan2f16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::atanf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::atanhf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::atanpif16(0.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::cosf16(0.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::coshf16(0.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::cospif16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::erff16(0.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::erfcf16(0.0f));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::exp10f16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::exp10m1f16(0.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::exp2f16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::exp2m1f16(0.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::expf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::expm1f16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::fmaf16(0.0f16, 0.0f16, 0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::hypotf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::logf16(1.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::sinhf16(0.0f16));

  ASSERT_FP_EQ(float16(8 << 5), LIBC_NAMESPACE::shared::ldexpf16(8.0f16, 5));
  ASSERT_FP_EQ(float16(-1 * (8 << 5)),
               LIBC_NAMESPACE::shared::ldexpf16(-8.0f16, 5));

  EXPECT_FP_EQ_ALL_ROUNDING(
      0.75f16, LIBC_NAMESPACE::shared::frexpf16(24.0f16, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbf16(1.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::log10f16(10.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::log10p1f16(9.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::log2f16(2.0f16));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::log2p1f16(1.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::logbf16(1.0f16));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbf16(1.0f16));

  EXPECT_FP_EQ(0x1.921fb6p+0f16, LIBC_NAMESPACE::shared::acosf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::sinf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::tanf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::sinpif16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::tanhf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::tanpif16(0.0f16));

  float16 canonicalizef16_cx = 0.0f16;
  float16 canonicalizef16_x = 0.0f16;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizef16(&canonicalizef16_cx,
                                                       &canonicalizef16_x));
  EXPECT_FP_EQ(0.0f16, canonicalizef16_cx);

  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::ceilf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::copysignf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::fabsf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::fdimf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::floorf16(0.0f16));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::fmaxf16(0.0f16, 0.0f16));

  float16 getpayloadf16_x = 0.0f16;
  EXPECT_FP_EQ(-1.0f16,
               LIBC_NAMESPACE::shared::getpayloadf16(&getpayloadf16_x));

  float16 setpayloadf16_res = 0.0f16;
  EXPECT_EQ(0,
            LIBC_NAMESPACE::shared::setpayloadf16(&setpayloadf16_res, 0.0f16));

  float16 setpayloadsigf16_res = 0.0f16;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::setpayloadsigf16(&setpayloadsigf16_res,
                                                        0.0f16));
  EXPECT_FP_EQ(0.0f16, setpayloadsigf16_res);
  float16 neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(neg_min_denormal, LIBC_NAMESPACE::shared::nextdownf16(0.0f16));
  float16 min_denormal = FPBits::min_subnormal(Sign ::POS).get_val();
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::shared::nextupf16(0.0f16));

  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::nextafterf16(0.0f16, 0.0f16));

#ifndef LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::nexttowardf16(0.0f16, 0.0L));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE

  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::fmaximumf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::fminimumf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::fminf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16,
               LIBC_NAMESPACE::shared::fmaximum_numf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16,
               LIBC_NAMESPACE::shared::fminimum_numf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::ufromfpf16(0.0f16, 0, 32));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::ufromfpxf16(0.0f16, 0, 32));
  EXPECT_FP_EQ(0x0p+0f16,
               LIBC_NAMESPACE::shared::fmaximum_magf16(0.0f16, 0.0f16));
  EXPECT_FP_EQ(0x0p+0f16,
               LIBC_NAMESPACE::shared::fminimum_magf16(0.0f16, 0.0f16));
  float16 totalorderf16_x = 0.0f16;
  float16 totalorderf16_y = 0.0f16;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalorderf16(&totalorderf16_x,
                                                     &totalorderf16_y));
  float16 totalordermagf16_x = 0.0f16;
  float16 totalordermagf16_y = 0.0f16;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalordermagf16(&totalordermagf16_x,
                                                        &totalordermagf16_y));
  EXPECT_FP_EQ(0x0p+0f16, LIBC_NAMESPACE::shared::fmodf16(1.0f16, 1.0f16));
  float16 modff16_iptr = 0.0f16;
  EXPECT_FP_EQ(0x0p+0f16,
               LIBC_NAMESPACE::shared::modff16(0.0f16, &modff16_iptr));
  EXPECT_FP_EQ(0.0f16, modff16_iptr);
}

#endif // LIBC_TYPES_HAS_FLOAT16

TEST(LlvmLibcSharedMathTest, AllFloat) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float>;
  int exponent;

  EXPECT_FP_EQ(0x1.921fb6p+0, LIBC_NAMESPACE::shared::acosf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::acoshf(1.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::acospif(1.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::asinf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::asinhf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::asinpif(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::atan2f(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::atanf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::atanhf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::cbrtf(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::cosf(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::coshf(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::cospif(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::exp10m1f(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::erff(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::exp10f(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::exp2m1f(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::expf(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::exp2f(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::expm1f(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fmaf(0.0f, 0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::hypotf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::logf(1.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::sinhf(0.0f));

  EXPECT_FP_EQ_ALL_ROUNDING(0.75f,
                            LIBC_NAMESPACE::shared::frexpf(24.0f, &exponent));
  EXPECT_EQ(exponent, 5);

  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbf(1.0f));

  ASSERT_FP_EQ(float(8 << 5), LIBC_NAMESPACE::shared::ldexpf(8.0f, 5));
  ASSERT_FP_EQ(float(-1 * (8 << 5)), LIBC_NAMESPACE::shared::ldexpf(-8.0f, 5));

  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbf(1.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::log1pf(0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::log10f(10.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::log2f(2.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::logbf(1.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::powf(0.0f, 0.0f));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::rsqrtf(1.0f));

  float s, c;
  LIBC_NAMESPACE::shared::sincosf(0.0f, &s, &c);
  ASSERT_FP_EQ(1.0f, c);
  ASSERT_FP_EQ(0.0f, s);
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::sinpif(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::sinf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::sqrtf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::tanf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::tanhf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::tanpif(0.0f));

  float canonicalizef_cx = 0.0f;
  float canonicalizef_x = 0.0f;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizef(&canonicalizef_cx,
                                                     &canonicalizef_x));
  EXPECT_FP_EQ(0.0f, canonicalizef_cx);

  EXPECT_FP_EQ(bfloat16(5.0f), LIBC_NAMESPACE::shared::bf16addf(2.0f, 3.0f));
  EXPECT_FP_EQ(bfloat16(2.0f), LIBC_NAMESPACE::shared::bf16divf(4.0f, 2.0f));
  EXPECT_FP_EQ(bfloat16(0.0f), LIBC_NAMESPACE::shared::bf16mulf(0.0f, 0.0f));
  EXPECT_FP_EQ(bfloat16(0.0f), LIBC_NAMESPACE::shared::bf16subf(0.0f, 0.0f));
  EXPECT_FP_EQ(bfloat16(10.0f),
               LIBC_NAMESPACE::shared::bf16fmaf(2.0f, 3.0f, 4.0f));

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::ceilf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::copysignf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fabsf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fdimf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::floorf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fmaxf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fmaximum_mag_numf(0.0f, 0.0f));

  float getpayloadf_x = 0.0f;
  EXPECT_FP_EQ(-1.0f, LIBC_NAMESPACE::shared::getpayloadf(&getpayloadf_x));

  float setpayloadf_res = 0.0f;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::setpayloadf(&setpayloadf_res, 0.0f));

  float setpayloadsigf_res = 0.0f;
  EXPECT_EQ(1,
            LIBC_NAMESPACE::shared::setpayloadsigf(&setpayloadsigf_res, 0.0f));
  EXPECT_FP_EQ(0.0f, setpayloadsigf_res);
  float neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(neg_min_denormal, LIBC_NAMESPACE::shared::nextdownf(0.0f));
  float min_denormal = FPBits::min_subnormal(Sign ::POS).get_val();
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::shared::nextupf(0.0f));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::nextafterf(0.0f, 0.0f));

#ifndef LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::nexttowardf(0.0f, 0.0L));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE

#ifdef LIBC_TYPES_HAS_FLOAT16
  EXPECT_FP_EQ(5.0f16, LIBC_NAMESPACE::shared::f16addf(2.0f, 3.0f));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16divf(0.0f, 1.0f));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16mulf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16subf(0.0f, 0.0f));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16sqrtf(0.0f));
  EXPECT_FP_EQ(10.0f16, LIBC_NAMESPACE::shared::f16fmaf(2.0f, 3.0f, 4.0f));
#endif // LIBC_TYPES_HAS_FLOAT16

  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fmaximumf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fminimumf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fminf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fmaximum_numf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fminimum_numf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::ufromfpf(0.0f, 0, 32));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::ufromfpxf(0.0f, 0, 32));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fmaximum_magf(0.0f, 0.0f));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fminimum_magf(0.0f, 0.0f));
  float totalorderf_x = 0.0f;
  float totalorderf_y = 0.0f;
  EXPECT_EQ(
      1, LIBC_NAMESPACE::shared::totalorderf(&totalorderf_x, &totalorderf_y));
  float totalordermagf_x = 0.0f;
  float totalordermagf_y = 0.0f;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalordermagf(&totalordermagf_x,
                                                      &totalordermagf_y));
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::fmodf(1.0f, 1.0f));
  float modff_iptr = 0.0f;
  EXPECT_FP_EQ(0x0p+0f, LIBC_NAMESPACE::shared::modff(0.0f, &modff_iptr));
  EXPECT_FP_EQ(0.0f, modff_iptr);
}

TEST(LlvmLibcSharedMathTest, AllDouble) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<double>;

  double sin, cos;
  LIBC_NAMESPACE::shared::sincos(0.0, &sin, &cos);

  EXPECT_FP_EQ(0x1.921fb54442d18p+0, LIBC_NAMESPACE::shared::acos(0.0));
  EXPECT_FP_EQ(0., LIBC_NAMESPACE::shared::asin(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::asinpi(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::atan(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::atan2(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::cbrt(0.0));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::cos(0.0));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::exp(0.0));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::exp2(0.0));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::exp10(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::expm1(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fma(0.0, 0.0, 0.0));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::ffma(0.0, 0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::hypot(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fsqrt(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::log(1.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::log10(1.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::log1p(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::log2(1.0));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::pow(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::sin(0.0));
  EXPECT_FP_EQ(1.0, cos);
  EXPECT_FP_EQ(0.0, sin);
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::sqrt(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::tan(0.0));
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogb(1.0));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogb(1.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::logb(1.0));

  double canonicalize_cx = 0.0;
  double canonicalize_x = 0.0;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalize(&canonicalize_cx,
                                                    &canonicalize_x));
  EXPECT_FP_EQ(0.0, canonicalize_cx);

  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16add(2.0, 3.0));
  EXPECT_FP_EQ(bfloat16(2.0), LIBC_NAMESPACE::shared::bf16div(4.0, 2.0));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16mul(0.0, 0.0));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16sub(0.0, 0.0));
  EXPECT_FP_EQ(bfloat16(10.0), LIBC_NAMESPACE::shared::bf16fma(2.0, 3.0, 4.0));

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::ceil(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::copysign(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fabs(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fadd(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fdim(0.0, 0.0));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::fdiv(1.0, 1.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::floor(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmax(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmaximum_mag_num(0.0, 0.0));

  double getpayload_x = 0.0;
  EXPECT_FP_EQ(-1.0, LIBC_NAMESPACE::shared::getpayload(&getpayload_x));

  double setpayload_res = 0.0;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::setpayload(&setpayload_res, 0.0));

  double setpayloadsig_res = 0.0;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::setpayloadsig(&setpayloadsig_res, 0.0));
  EXPECT_FP_EQ(0.0, setpayloadsig_res);
  double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(neg_min_denormal, LIBC_NAMESPACE::shared::nextdown(0.0));
  double min_denormal = FPBits::min_subnormal(Sign ::POS).get_val();
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::shared::nextup(0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::nextafter(0.0, 0.0));

#ifndef LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::nexttoward(0.0, 0.0L));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE

#ifdef LIBC_TYPES_HAS_FLOAT16
  EXPECT_FP_EQ(5.0f16, LIBC_NAMESPACE::shared::f16add(2.0, 3.0));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16div(0.0, 1.0));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16mul(0.0, 0.0));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16sub(0.0, 0.0));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16sqrt(0.0));
  EXPECT_FP_EQ(10.0f16, LIBC_NAMESPACE::shared::f16fma(2.0, 3.0, 4.0));
#endif // LIBC_TYPES_HAS_FLOAT16

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmaximum(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fminimum(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmin(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmaximum_num(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fminimum_num(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::ufromfp(0.0, 0, 32));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::ufromfpx(0.0, 0, 32));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmaximum_mag(0.0, 0.0));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fminimum_mag(0.0, 0.0));
  double totalorder_x = 0.0;
  double totalorder_y = 0.0;
  EXPECT_EQ(1,
            LIBC_NAMESPACE::shared::totalorder(&totalorder_x, &totalorder_y));
  double totalordermag_x = 0.0;
  double totalordermag_y = 0.0;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalordermag(&totalordermag_x,
                                                     &totalordermag_y));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::fmod(1.0, 1.0));
  double modf_iptr = 0.0;
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::modf(0.0, &modf_iptr));
  EXPECT_FP_EQ(0.0, modf_iptr);
}

// TODO: Enable the tests when double-double type is supported.
#ifndef LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE

TEST(LlvmLibcSharedMathTest, AllLongDouble) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<long double>;
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::dfmal(0.0L, 0.0L, 0.0L));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fsqrtl(0.0L));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::dsqrtl(0.0));
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbl(1.0L));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbl(1.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::logbl(1.0L));
  EXPECT_FP_EQ(10.0f, LIBC_NAMESPACE::shared::ffmal(2.0L, 3.0, 4.0L));

  long double canonicalizel_cx = 0.0L;
  long double canonicalizel_x = 0.0L;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizel(&canonicalizel_cx,
                                                     &canonicalizel_x));
  EXPECT_FP_EQ(0.0L, canonicalizel_cx);

  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16addl(2.0L, 3.0L));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16subl(0.0L, 0.0L));
  EXPECT_FP_EQ(bfloat16(2.0), LIBC_NAMESPACE::shared::bf16divl(6.0L, 3.0L));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16mull(0.0L, 0.0L));
  EXPECT_FP_EQ(bfloat16(10.0),
               LIBC_NAMESPACE::shared::bf16fmal(2.0L, 3.0L, 4.0L));

  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::ceill(0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::copysignl(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::daddl(0.0L, 0.0L));
  EXPECT_FP_EQ(1.0, LIBC_NAMESPACE::shared::ddivl(1.0L, 1.0L));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::dmull(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::dsubl(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::fabsl(0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::faddl(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::fdiml(0.0L, 0.0L));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::fdivl(1.0L, 1.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::floorl(0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::fmaxl(0.0L, 0.0L));
  long double getpayloadl_x = 0.0L;
  EXPECT_FP_EQ(-1.0L, LIBC_NAMESPACE::shared::getpayloadl(&getpayloadl_x));

  long double setpayloadl_res = 0.0L;
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::setpayloadl(&setpayloadl_res, 0.0L));

  long double setpayloadsigl_res = 0.0L;
  EXPECT_EQ(1,
            LIBC_NAMESPACE::shared::setpayloadsigl(&setpayloadsigl_res, 0.0L));
  EXPECT_FP_EQ(0.0L, setpayloadsigl_res);

  long double neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(neg_min_denormal, LIBC_NAMESPACE::shared::nextdownl(0.0L));
  long double min_denormal = FPBits::min_subnormal(Sign ::POS).get_val();
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::shared::nextupl(0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::nexttowardl(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::nextafterl(0.0L, 0.0L));

#ifdef LIBC_TYPES_HAS_FLOAT16
  EXPECT_FP_EQ(5.0f16, LIBC_NAMESPACE::shared::f16addl(2.0L, 3.0L));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16divl(0.0L, 1.0L));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16mull(0.0L, 0.0L));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16subl(0.0L, 0.0L));
  EXPECT_FP_EQ(1.0f16, LIBC_NAMESPACE::shared::f16sqrtl(1.0L));
  EXPECT_FP_EQ(10.0f16, LIBC_NAMESPACE::shared::f16fmal(2.0L, 3.0L, 4.0L));
#endif // LIBC_TYPES_HAS_FLOAT16

  EXPECT_FP_EQ(0.0L, LIBC_NAMESPACE::shared::sqrtl(0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fmaximuml(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fminimuml(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fminl(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fmaximum_numl(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fminimum_numl(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::ufromfpl(0.0L, 0, 32));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::ufromfpxl(0.0L, 0, 32));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fmaximum_magl(0.0L, 0.0L));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fminimum_magl(0.0L, 0.0L));
  long double totalorderl_x = 0.0L;
  long double totalorderl_y = 0.0L;
  EXPECT_EQ(
      1, LIBC_NAMESPACE::shared::totalorderl(&totalorderl_x, &totalorderl_y));
  long double totalordermagl_x = 0.0L;
  long double totalordermagl_y = 0.0L;
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalordermagl(&totalordermagl_x,
                                                      &totalordermagl_y));
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::fmodl(1.0L, 1.0L));
  long double modfl_iptr = 0.0L;
  EXPECT_FP_EQ(0x0p+0L, LIBC_NAMESPACE::shared::modfl(0.0L, &modfl_iptr));
  EXPECT_FP_EQ(0.0L, modfl_iptr);
}

#endif // LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedMathTest, AllFloat128) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float128>;
  int exponent;

  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::atan2f128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::ffmaf128(
                         float128(0.0), float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::shared::fsqrtf128(float128(1.0f)));
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
  EXPECT_FP_EQ(float128(1.0), LIBC_NAMESPACE::shared::sqrtf128(float128(1.0)));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::shared::dsqrtf128(float128(0.0)));

  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbf128(float128(1.0)));

  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16addf128(
                                  float128(2.0), float128(3.0)));

  float128 canonicalizef128_cx = float128(0.0);
  float128 canonicalizef128_x = float128(0.0);
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizef128(&canonicalizef128_cx,
                                                        &canonicalizef128_x));
  EXPECT_FP_EQ(float128(0.0), canonicalizef128_cx);

  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16subf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16fmaf128(
                                  float128(0.0), float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::bf16mulf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(bfloat16(2.0), LIBC_NAMESPACE::shared::bf16divf128(
                                  float128(4.0), float128(2.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::ceilf128(float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::copysignf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::daddf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(1.0,
               LIBC_NAMESPACE::shared::ddivf128(float128(1.0), float128(1.0)));
  EXPECT_FP_EQ(0.0,
               LIBC_NAMESPACE::shared::dmulf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(0.0,
               LIBC_NAMESPACE::shared::dsubf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fabsf128(float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::faddf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::floorf128(float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::fdimf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(1.0f,
               LIBC_NAMESPACE::shared::fdivf128(float128(1.0), float128(1.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::fmaxf128(float128(0.0), float128(0.0)));

  float128 getpayloadf128_x = float128(0.0);
  EXPECT_FP_EQ(float128(-1.0),
               LIBC_NAMESPACE::shared::getpayloadf128(&getpayloadf128_x));

  float128 setpayloadf128_res = float128(0.0);
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::setpayloadf128(&setpayloadf128_res,
                                                      float128(0.0)));

  float128 setpayloadsigf128_res = float128(0.0);
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::setpayloadsigf128(&setpayloadsigf128_res,
                                                         float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), setpayloadsigf128_res);

  float128 neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(neg_min_denormal,
               LIBC_NAMESPACE::shared::nextdownf128(float128(0.0)));
  float128 min_denormal = FPBits::min_subnormal(Sign ::POS).get_val();
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::shared::nextupf128(float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::nextafterf128(
                                  float128(0.0), float128(0.0)));

#ifdef LIBC_TYPES_HAS_FLOAT16
  EXPECT_FP_EQ(10.0f16, LIBC_NAMESPACE::shared::f16fmaf128(
                            float128(2.0), float128(3.0), float128(4.0)));
  EXPECT_FP_EQ(
      5.0f16, LIBC_NAMESPACE::shared::f16addf128(float128(2.0), float128(3.0)));
  EXPECT_FP_EQ(
      0.0f16, LIBC_NAMESPACE::shared::f16divf128(float128(0.0), float128(1.0)));

  EXPECT_FP_EQ(
      0.0f16, LIBC_NAMESPACE::shared::f16mulf128(float128(0.0), float128(0.0)));

  EXPECT_FP_EQ(
      0.0f16, LIBC_NAMESPACE::shared::f16subf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(0.0f16, LIBC_NAMESPACE::shared::f16sqrtf128(float128(0.0)));
#endif // LIBC_TYPES_HAS_FLOAT16

  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fmaximumf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fminimumf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::fminf128(float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fmaximum_numf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fminimum_numf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::ufromfpf128(float128(0.0), 0, 32));
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::ufromfpxf128(float128(0.0), 0, 32));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fmaximum_magf128(
                                  float128(0.0), float128(0.0)));
  EXPECT_FP_EQ(float128(0.0), LIBC_NAMESPACE::shared::fminimum_magf128(
                                  float128(0.0), float128(0.0)));
  float128 totalorderf128_x = float128(0.0);
  float128 totalorderf128_y = float128(0.0);
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalorderf128(&totalorderf128_x,
                                                      &totalorderf128_y));
  float128 totalordermagf128_x = float128(0.0);
  float128 totalordermagf128_y = float128(0.0);
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalordermagf128(&totalordermagf128_x,
                                                         &totalordermagf128_y));
  LIBC_NAMESPACE::shared::fmodf128(float128(1.0), float128(1.0));
  float128 modff128_iptr = float128(0.0);
  EXPECT_FP_EQ(float128(0.0),
               LIBC_NAMESPACE::shared::modff128(float128(0.0), &modff128_iptr));
  EXPECT_FP_EQ(float128(0.0), modff128_iptr);
}

#endif // LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedMathTest, AllBFloat16) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::atanbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::asinbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(5.0), LIBC_NAMESPACE::shared::bf16add(2.0, 3.0));
  EXPECT_FP_EQ(bfloat16(2.0f), LIBC_NAMESPACE::shared::bf16divf(4.0f, 2.0f));
  EXPECT_FP_EQ(bfloat16(2.0), LIBC_NAMESPACE::shared::bf16div(4.0, 2.0));

  bfloat16 canonicalizebf16_cx = bfloat16(0.0);
  bfloat16 canonicalizebf16_x = bfloat16(0.0);
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::canonicalizebf16(&canonicalizebf16_cx,
                                                        &canonicalizebf16_x));
  EXPECT_FP_EQ(bfloat16(0.0), canonicalizebf16_cx);
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::cbrtbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::ceilbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::copysignbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fabsbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::floorbf16(bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::fdimbf16(bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(10.0),
               LIBC_NAMESPACE::shared::fmabf16(bfloat16(2.0), bfloat16(3.0),
                                               bfloat16(4.0)));
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::fmaxbf16(bfloat16(0.0), bfloat16(0.0)));

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::shared::fmaximum_mag_numbf16(
                         bfloat16(0.0), bfloat16(0.0)));

  bfloat16 getpayloadbf16_x = bfloat16(0.0);
  EXPECT_FP_EQ(bfloat16(-1.0),
               LIBC_NAMESPACE::shared::getpayloadbf16(&getpayloadbf16_x));

  EXPECT_FP_EQ(bfloat16(5.0),
               LIBC_NAMESPACE::shared::hypotbf16(bfloat16(4.0), bfloat16(3.0)));

  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::logbbf16(bfloat16(1.0f)));

  bfloat16 setpayloadbf16_res = bfloat16(0.0);
  EXPECT_EQ(0, LIBC_NAMESPACE::shared::setpayloadbf16(&setpayloadbf16_res,
                                                      bfloat16(0.0)));

  bfloat16 setpayloadsigbf16_res = bfloat16(0.0);
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::setpayloadsigbf16(&setpayloadsigbf16_res,
                                                         bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), setpayloadsigbf16_res);

  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::log_bf16(bfloat16(1.0)));

  bfloat16 neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();
  EXPECT_FP_EQ(neg_min_denormal,
               LIBC_NAMESPACE::shared::nextdownbf16(bfloat16(0.0)));
  bfloat16 min_denormal = FPBits::min_subnormal(Sign ::POS).get_val();
  EXPECT_FP_EQ(min_denormal, LIBC_NAMESPACE::shared::nextupbf16(bfloat16(0.0)));

  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::nextafterbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(1.0), LIBC_NAMESPACE::shared::sqrtbf16(bfloat16(1.0)));
#ifndef LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::nexttowardbf16(bfloat16(0.0), 0.0L));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_DOUBLE_DOUBLE

  EXPECT_EQ(0, LIBC_NAMESPACE::shared::ilogbbf16(bfloat16(1.0)));
  EXPECT_EQ(0L, LIBC_NAMESPACE::shared::llogbbf16(bfloat16(1.0)));

  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fmaximumbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fminimumbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::fminbf16(bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fmaximum_numbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fminimum_numbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::ufromfpbf16(bfloat16(0.0), 0, 32));
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::ufromfpxbf16(bfloat16(0.0), 0, 32));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fmaximum_magbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  EXPECT_FP_EQ(bfloat16(0.0), LIBC_NAMESPACE::shared::fminimum_magbf16(
                                  bfloat16(0.0), bfloat16(0.0)));
  bfloat16 totalorderbf16_x = bfloat16(0.0);
  bfloat16 totalorderbf16_y = bfloat16(0.0);
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalorderbf16(&totalorderbf16_x,
                                                      &totalorderbf16_y));
  bfloat16 totalordermagbf16_x = bfloat16(0.0);
  bfloat16 totalordermagbf16_y = bfloat16(0.0);
  EXPECT_EQ(1, LIBC_NAMESPACE::shared::totalordermagbf16(&totalordermagbf16_x,
                                                         &totalordermagbf16_y));
  LIBC_NAMESPACE::shared::fmodbf16(bfloat16(1.0), bfloat16(1.0));
  bfloat16 modfbf16_iptr = bfloat16(0.0);
  EXPECT_FP_EQ(bfloat16(0.0),
               LIBC_NAMESPACE::shared::modfbf16(bfloat16(0.0), &modfbf16_iptr));
  EXPECT_FP_EQ(bfloat16(0.0), modfbf16_iptr);
}
