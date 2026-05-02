//===-- Unittests for shared math functions in constexpr context ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define LIBC_ENABLE_CONSTEXPR 1

#include "shared/math.h"
#include "test/UnitTest/Test.h"

//===----------------------------------------------------------------------===//
//                       Double Tests
//===----------------------------------------------------------------------===//

static_assert(0.0 == LIBC_NAMESPACE::shared::ceil(0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::copysign(0.0, 0.0));
static_assert(1.0 == LIBC_NAMESPACE::shared::fabs(-1.0));
static_assert(1.0 == LIBC_NAMESPACE::shared::fdim(1.0, 0.0));
static_assert(0.0f == LIBC_NAMESPACE::shared::fdiv(0.0, 1.0));
static_assert(1.0 == LIBC_NAMESPACE::shared::floor(1.2));
static_assert(2.0 == LIBC_NAMESPACE::shared::fmaximum_mag_num(1.0, 2.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::log(1.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmaximum(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fminimum(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmin(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmax(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmaximum_num(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fminimum_num(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fromfp(0.0, 0, 32));
static_assert(0.0 == LIBC_NAMESPACE::shared::fromfpx(0.0, 0, 32));
static_assert(0.0 == LIBC_NAMESPACE::shared::ufromfp(0.0, 0, 32));
static_assert(0.0 == LIBC_NAMESPACE::shared::ufromfpx(0.0, 0, 32));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmaximum_mag(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fminimum_mag(0.0, 0.0));

constexpr double TOTALORDER_X = 0.0;
constexpr double TOTALORDER_Y = 0.0;
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalorder(&TOTALORDER_X, &TOTALORDER_Y));
constexpr double TOTALORDERMAG_X = 0.0;
constexpr double TOTALORDERMAG_Y = 0.0;
static_assert(1 == LIBC_NAMESPACE::shared::totalordermag(&TOTALORDERMAG_X,
                                                         &TOTALORDERMAG_Y));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmod(4.0, 2.0));
static_assert(0.0 == [] {
  double iptr = 0;
  return LIBC_NAMESPACE::shared::modf(0, &iptr);
}());
static_assert(0.0 == LIBC_NAMESPACE::shared::fminimum_mag_num(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::remainder(1.0, 1.0));
static_assert(0.0 == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::remquo(1.0, 1.0, &exp);
}());
static_assert(0.0 == LIBC_NAMESPACE::shared::ldexp(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::scalbln(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::scalbn(0.0, 0.0));
static_assert(0.0 == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::frexp(0.0, &exp);
}());

//===----------------------------------------------------------------------===//
//                       Float Tests
//===----------------------------------------------------------------------===//

static_assert(0.0f == LIBC_NAMESPACE::shared::ceilf(0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::copysignf(0.0f, 0.0f));
static_assert(1.0f == LIBC_NAMESPACE::shared::fabsf(-1.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fadd(0.0, 0.0));
static_assert(1.0f == LIBC_NAMESPACE::shared::fdimf(1.0f, 0.0f));
static_assert(2.0f == LIBC_NAMESPACE::shared::fmaximum_mag_numf(1.0f, 2.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::floorf(0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaximumf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimumf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaxf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaximum_numf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimum_numf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fromfp(0.0f, 0, 32));
static_assert(0.0f == LIBC_NAMESPACE::shared::fromfpx(0.0f, 0, 32));
static_assert(0.0f == LIBC_NAMESPACE::shared::ufromfpf(0.0f, 0, 32));
static_assert(0.0f == LIBC_NAMESPACE::shared::ufromfpxf(0.0f, 0, 32));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaximum_magf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimum_magf(0.0f, 0.0f));

constexpr float TOTALORDERF_X = 0.0f;
constexpr float TOTALORDERF_Y = 0.0f;
static_assert(1 == LIBC_NAMESPACE::shared::totalorderf(&TOTALORDERF_X,
                                                       &TOTALORDERF_Y));
constexpr float TOTALORDERMAGF_X = 0.0f;
constexpr float TOTALORDERMAGF_Y = 0.0f;
static_assert(1 == LIBC_NAMESPACE::shared::totalordermagf(&TOTALORDERMAGF_X,
                                                          &TOTALORDERMAGF_Y));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmodf(4.0f, 2.0f));
static_assert(0.0f == [] {
  float iptr = 0.0f;
  return LIBC_NAMESPACE::shared::modff(0.0f, &iptr);
}());
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimum_mag_numf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::remainderf(1.0f, 1.0f));
static_assert(0.0f == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::remquof(1.0f, 1.0f, &exp);
}());
static_assert(0.0f == LIBC_NAMESPACE::shared::scalblnf(0.0f, 0.0));
static_assert(0.0f == LIBC_NAMESPACE::shared::scalbnf(0.0f, 0.0));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmul(0.0, 0.0));
static_assert(0.0f == LIBC_NAMESPACE::shared::fsub(0.0, 0.0));
//===----------------------------------------------------------------------===//
//                       Float16 Tests
//===----------------------------------------------------------------------===//

#ifdef LIBC_TYPES_HAS_FLOAT16

static_assert(0.0f16 == LIBC_NAMESPACE::shared::ceilf16(0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::copysignf16(0.0f16, 0.0f16));
static_assert(1.0f16 == LIBC_NAMESPACE::shared::fabsf16(-1.0f16));
static_assert(1.0f16 == LIBC_NAMESPACE::shared::fdimf16(1.0f16, 0.0f16));
static_assert(3.0f16 == LIBC_NAMESPACE::shared::floorf16(3.7f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fmaximumf16(0.0f16, 0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fminimumf16(0.0f16, 0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fminf16(0.0f16, 0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fmaxf16(0.0f16, 0.0f16));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fmaximum_numf16(0.0f16, 0.0f16));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fminimum_numf16(0.0f16, 0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fromfpf16(0.0f16, 0, 32));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fromfpxf16(0.0f16, 0, 32));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::ufromfpf16(0.0f16, 0, 32));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::ufromfpxf16(0.0f16, 0, 32));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fmaximum_magf16(0.0f16, 0.0f16));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fminimum_magf16(0.0f16, 0.0f16));

constexpr float16 TOTALORDERF16_X = 0.0f16;
constexpr float16 TOTALORDERF16_Y = 0.0f16;
static_assert(1 == LIBC_NAMESPACE::shared::totalorderf16(&TOTALORDERF16_X,
                                                         &TOTALORDERF16_Y));

constexpr float16 TOTALORDERMAGF16_X = 0.0f16;
constexpr float16 TOTALORDERMAGF16_Y = 0.0f16;
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalordermagf16(&TOTALORDERMAGF16_X,
                                                       &TOTALORDERMAGF16_Y));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::fmodf16(4.0f16, 2.0f16));
static_assert(0.0f16 == [] {
  float16 iptr{};
  return LIBC_NAMESPACE::shared::modff16(0.0f16, &iptr);
}());
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fmaximum_mag_numf16(0.0f16, 0.0f16));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fminimum_mag_numf16(0.0f16, 0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::remainderf16(1.0f16, 1.0f16));
static_assert(0.0f16 == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::remquof16(1.0f16, 1.0f16, &exp);
}());
static_assert(0.0f16 == LIBC_NAMESPACE::shared::scalblnf16(0.0f16, 0.0));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::scalbnf16(0.0f16, 0.0));
#endif // LIBC_TYPES_HAS_FLOAT16

//===----------------------------------------------------------------------===//
//                       Long Double Tests
//===----------------------------------------------------------------------===//

// TODO(issue#185232): Mark as constexpr once the refactor is done.
#if 0 // Temporarily disable long double tests

static_assert(0.0L == LIBC_NAMESPACE::shared::ceill(0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::copysignl(0.0L, 0.0L));
static_assert(0.0 == LIBC_NAMESPACE::shared::ddivl(0.0L, 1.0L));
static_assert(0.0 == LIBC_NAMESPACE::shared::dmull(0.0L, 1.0L));
static_assert(1.0L == LIBC_NAMESPACE::shared::fabsl(-1.0L));
static_assert(0.0f == LIBC_NAMESPACE::shared::faddl(0.0L, 0.0L));
static_assert(1.0L == LIBC_NAMESPACE::shared::fdiml(1.0L, 0.0L));
static_assert(0.0f == LIBC_NAMESPACE::shared::fdivl(0.0L, 1.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::floorl(0.0L));
static_assert(bfloat16(0.0) == LIBC_NAMESPACE::shared::bf16subl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::sqrtl(0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fmaximuml(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fminimuml(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fminl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fmaxl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fmaximum_numl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fminimum_numl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fromfpl(0.0L, 0, 32));
static_assert(0.0L == LIBC_NAMESPACE::shared::fromfpxl(0.0L, 0, 32));
static_assert(0.0L == LIBC_NAMESPACE::shared::ufromfpl(0.0L, 0, 32));
static_assert(0.0L == LIBC_NAMESPACE::shared::ufromfpxl(0.0L, 0, 32));
static_assert(0.0L == LIBC_NAMESPACE::shared::fmaximum_magl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fminimum_magl(0.0L, 0.0L));

constexpr long double TOTALORDERL_X = 0.0L;
constexpr long double TOTALORDERL_Y = 0.0L;
static_assert(1 == LIBC_NAMESPACE::shared::totalorderl(&TOTALORDERL_X,
                                                       &TOTALORDERL_Y));
constexpr long double TOTALORDERMAGL_X = 0.0L;
constexpr long double TOTALORDERMAGL_Y = 0.0L;
static_assert(1 == LIBC_NAMESPACE::shared::totalordermagl(&TOTALORDERMAGL_X,
                                                          &TOTALORDERMAGL_Y));
static_assert(0.0L == LIBC_NAMESPACE::shared::fmodl(4.0L, 2.0L));

static_assert(0.0L == [] {
  long double iptr{};
  return LIBC_NAMESPACE::shared::modfl(0.0L, &iptr);
}());
static_assert(0.0L == LIBC_NAMESPACE::shared::fmaximum_mag_numl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fminimum_mag_numl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::remainderl(1.0L, 1.0L));
static_assert(0.0L == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::remquol(1.0L, 1.0L, &exp);
}());
static_assert(0.0L == LIBC_NAMESPACE::shared::ldexpl(0.0L, 0.0));
static_assert(0.0L == LIBC_NAMESPACE::shared::scalblnl(0.0L, 0.0));
static_assert(0.0L == LIBC_NAMESPACE::shared::scalbnl(0.0L, 0.0));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmull(0.0L, 0.0L));
static_assert(0.0f == LIBC_NAMESPACE::shared::fsubl(0.0L, 0.0L));
static_assert(0.0L == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::frexpl(0.0L, &exp);
}());

#endif

//===----------------------------------------------------------------------===//
//                       Float128 Tests
//===----------------------------------------------------------------------===//

#ifdef LIBC_TYPES_HAS_FLOAT128

static_assert(float128(0.0) == LIBC_NAMESPACE::shared::ceilf128(float128(0.0)));
static_assert(float128(1.0) ==
              LIBC_NAMESPACE::shared::fabsf128(float128(-1.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::copysignf128(float128(0.0),
                                                   float128(0.0)));
static_assert(0.0 ==
              LIBC_NAMESPACE::shared::ddivf128(float128(0.0), float128(1.0)));
static_assert(0.0 ==
              LIBC_NAMESPACE::shared::dmulf128(float128(0.0), float128(1.0)));
static_assert(0.0f ==
              LIBC_NAMESPACE::shared::faddf128(float128(0.0), float128(0.0)));
static_assert(float128(1.0) ==
              LIBC_NAMESPACE::shared::fdimf128(float128(1.0), float128(0.0)));
static_assert(0.0f ==
              LIBC_NAMESPACE::shared::fdivf128(float128(0.0), float128(1.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::floorf128(float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fmaximumf128(float128(0.0),
                                                   float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fminimumf128(float128(0.0),
                                                   float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fminf128(float128(0.0), float128(0.0)));
static_assert(0.0 == LIBC_NAMESPACE::shared::dsqrtf128(float128(0.0)));

static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fmaxf128(float128(0.0), float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_numf128(float128(0.0),
                                                       float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_numf128(float128(0.0),
                                                       float128(0.0)));

static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fromfpf128(float128(0.0), 0, 32));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fromfpxf128(float128(0.0), 0, 32));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::ufromfpf128(float128(0.0), 0, 32));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::ufromfpxf128(float128(0.0), 0, 32));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_magf128(float128(0.0),
                                                       float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_magf128(float128(0.0),
                                                       float128(0.0)));
constexpr float128 TOTALORDERF128_X = float128(0.0);
constexpr float128 TOTALORDERF128_Y = float128(0.0);
static_assert(1 == LIBC_NAMESPACE::shared::totalorderf128(&TOTALORDERF128_X,
                                                          &TOTALORDERF128_Y));
constexpr float128 TOTALORDERMAGF128_X = float128(0.0);
constexpr float128 TOTALORDERMAGF128_Y = float128(0.0);
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalordermagf128(&TOTALORDERMAGF128_X,
                                                        &TOTALORDERMAGF128_Y));
static_assert(0 ==
              LIBC_NAMESPACE::shared::fmodf128(float128(4.0), float128(2.0)));
static_assert(float128(0.0) == [] {
  float128 iptr{};
  return LIBC_NAMESPACE::shared::modff128(float128(0.0), &iptr);
}());
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_mag_numf128(float128(0.0),
                                                           float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_mag_numf128(float128(0.0),
                                                           float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::remainderf128(float128(1.0),
                                                    float128(1.0)));
static_assert(float128(0.0) == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::remquof128(float128(1.0), float128(1.0), &exp);
}());
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::scalblnf128(float128(0.0), 0.0));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::scalbnf128(float128(0.0), 0.0));
static_assert(0.0f ==
              LIBC_NAMESPACE::shared::fmulf128(float128(0.0), float128(0.0)));
static_assert(0.0f ==
              LIBC_NAMESPACE::shared::fsubf128(float128(0.0), float128(0.0)));

#endif // LIBC_TYPES_HAS_FLOAT128

//===----------------------------------------------------------------------===//
//                       BFloat16 Tests
//===----------------------------------------------------------------------===//

static_assert(bfloat16(0.0) == LIBC_NAMESPACE::shared::asinbf16(bfloat16(0.0)));
static_assert(bfloat16(0.0) == LIBC_NAMESPACE::shared::ceilbf16(bfloat16(0.0)));
static_assert(bfloat16(1.0) ==
              LIBC_NAMESPACE::shared::fabsbf16(bfloat16(-1.0)));
static_assert(bfloat16(2.0) ==
              LIBC_NAMESPACE::shared::fmaximum_mag_numbf16(bfloat16(1.0),
                                                           bfloat16(2.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::copysignbf16(bfloat16(0.0),
                                                   bfloat16(0.0)));
static_assert(bfloat16(1.0) ==
              LIBC_NAMESPACE::shared::fdimbf16(bfloat16(1.0), bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::floorbf16(bfloat16(0.0f)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::logbbf16(bfloat16(1.0f)));
static_assert(0 == LIBC_NAMESPACE::shared::ilogbbf16(bfloat16(1.0)));
static_assert(0L == LIBC_NAMESPACE::shared::llogbbf16(bfloat16(1.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fmaximumbf16(bfloat16(0.0),
                                                   bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fminimumbf16(bfloat16(0.0),
                                                   bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fminbf16(bfloat16(0.0), bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fmaxbf16(bfloat16(0.0), bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_numbf16(bfloat16(0.0),
                                                       bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_numbf16(bfloat16(0.0),
                                                       bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fromfpbf16(bfloat16(0.0), 0, 32));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fromfpxbf16(bfloat16(0.0), 0, 32));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::ufromfpbf16(bfloat16(0.0), 0, 32));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::ufromfpxbf16(bfloat16(0.0), 0, 32));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_magbf16(bfloat16(0.0),
                                                       bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_magbf16(bfloat16(0.0),
                                                       bfloat16(0.0)));

constexpr bfloat16 TOTALORDERBF16_X = bfloat16(0.0);
constexpr bfloat16 TOTALORDERBF16_Y = bfloat16(0.0);
static_assert(1 == LIBC_NAMESPACE::shared::totalorderbf16(&TOTALORDERBF16_X,
                                                          &TOTALORDERBF16_Y));
constexpr bfloat16 TOTALORDERMAGBF16_X = bfloat16(0.0);
constexpr bfloat16 TOTALORDERMAGBF16_Y = bfloat16(0.0);
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalordermagbf16(&TOTALORDERMAGBF16_X,
                                                        &TOTALORDERMAGBF16_Y));
static_assert(0 ==
              LIBC_NAMESPACE::shared::fmodbf16(bfloat16(4.0), bfloat16(2.0)));
static_assert(bfloat16(0.0) == [] {
  bfloat16 iptr{};
  return LIBC_NAMESPACE::shared::modfbf16(bfloat16(0.0), &iptr);
}());
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_mag_numbf16(bfloat16(0.0),
                                                           bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::remainderbf16(bfloat16(1.0),
                                                    bfloat16(1.0)));
static_assert(bfloat16(0.0) == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::remquobf16(bfloat16(1.0), bfloat16(1.0), &exp);
}());
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::ldexpbf16(bfloat16(0.0), 0.0));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::scalblnbf16(bfloat16(0.0), 0.0));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::scalbnbf16(bfloat16(0.0), 0.0));
static_assert(bfloat16(0.0) == [] {
  int exp{};
  return LIBC_NAMESPACE::shared::frexpbf16(bfloat16(0.0), &exp);
}());

TEST(LlvmLibcSharedMathTest, ConstantEvaluation) {}
