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
static_assert(0.0 == LIBC_NAMESPACE::shared::ufromfp(0.0, 0, 32));
static_assert(0.0 == LIBC_NAMESPACE::shared::ufromfpx(0.0, 0, 32));
static_assert(0.0 == LIBC_NAMESPACE::shared::fmaximum_mag(0.0, 0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::fminimum_mag(0.0, 0.0));

constexpr double totalorder_x = 0.0;
constexpr double totalorder_y = 0.0;
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalorder(&totalorder_x, &totalorder_y));
constexpr double totalordermag_x = 0.0;
constexpr double totalordermag_y = 0.0;
static_assert(1 == LIBC_NAMESPACE::shared::totalordermag(&totalordermag_x,
                                                         &totalordermag_y));

//===----------------------------------------------------------------------===//
//                       Float Tests
//===----------------------------------------------------------------------===//

static_assert(0.0f == LIBC_NAMESPACE::shared::ceilf(0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::copysignf(0.0f, 0.0f));
static_assert(1.0f == LIBC_NAMESPACE::shared::fabsf(-1.0f));
static_assert(1.0f == LIBC_NAMESPACE::shared::fdimf(1.0f, 0.0f));
static_assert(2.0f == LIBC_NAMESPACE::shared::fmaximum_mag_numf(1.0f, 2.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::floorf(0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaximumf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimumf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaxf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaximum_numf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimum_numf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::ufromfpf(0.0f, 0, 32));
static_assert(0.0f == LIBC_NAMESPACE::shared::ufromfpxf(0.0f, 0, 32));
static_assert(0.0f == LIBC_NAMESPACE::shared::fmaximum_magf(0.0f, 0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::fminimum_magf(0.0f, 0.0f));

constexpr float totalorderf_x = 0.0f;
constexpr float totalorderf_y = 0.0f;
static_assert(1 == LIBC_NAMESPACE::shared::totalorderf(&totalorderf_x,
                                                       &totalorderf_y));
constexpr float totalordermagf_x = 0.0f;
constexpr float totalordermagf_y = 0.0f;
static_assert(1 == LIBC_NAMESPACE::shared::totalordermagf(&totalordermagf_x,
                                                          &totalordermagf_y));

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
static_assert(0.0f16 == LIBC_NAMESPACE::shared::ufromfpf16(0.0f16, 0, 32));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::ufromfpxf16(0.0f16, 0, 32));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fmaximum_magf16(0.0f16, 0.0f16));
static_assert(0.0f16 ==
              LIBC_NAMESPACE::shared::fminimum_magf16(0.0f16, 0.0f16));

constexpr float16 totalorderf16_x = 0.0f16;
constexpr float16 totalorderf16_y = 0.0f16;
static_assert(1 == LIBC_NAMESPACE::shared::totalorderf16(&totalorderf16_x,
                                                         &totalorderf16_y));

constexpr float16 totalordermagf16_x = 0.0f16;
constexpr float16 totalordermagf16_y = 0.0f16;
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalordermagf16(&totalordermagf16_x,
                                                       &totalordermagf16_y));

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
static_assert(0.0L == LIBC_NAMESPACE::shared::ufromfpl(0.0L, 0, 32));
static_assert(0.0L == LIBC_NAMESPACE::shared::ufromfpxl(0.0L, 0, 32));
static_assert(0.0L == LIBC_NAMESPACE::shared::fmaximum_magl(0.0L, 0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::fminimum_magl(0.0L, 0.0L));

constexpr long double totalorderl_x = 0.0L;
constexpr long double totalorderl_y = 0.0L;
static_assert(1 == LIBC_NAMESPACE::shared::totalorderl(&totalorderl_x,
                                                       &totalorderl_y));
constexpr long double totalordermagl_x = 0.0L;
constexpr long double totalordermagl_y = 0.0L;
static_assert(1 == LIBC_NAMESPACE::shared::totalordermagl(&totalordermagl_x,
                                                          &totalordermagl_y));

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
              LIBC_NAMESPACE::shared::ufromfpf128(float128(0.0), 0, 32));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::ufromfpxf128(float128(0.0), 0, 32));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_magf128(float128(0.0),
                                                       float128(0.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_magf128(float128(0.0),
                                                       float128(0.0)));
constexpr float128 totalorderf128_x = float128(0.0);
constexpr float128 totalorderf128_y = float128(0.0);
static_assert(1 == LIBC_NAMESPACE::shared::totalorderf128(&totalorderf128_x,
                                                          &totalorderf128_y));
constexpr float128 totalordermagf128_x = float128(0.0);
constexpr float128 totalordermagf128_y = float128(0.0);
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalordermagf128(&totalordermagf128_x,
                                                        &totalordermagf128_y));

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
              LIBC_NAMESPACE::shared::ufromfpbf16(bfloat16(0.0), 0, 32));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::ufromfpxbf16(bfloat16(0.0), 0, 32));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fmaximum_magbf16(bfloat16(0.0),
                                                       bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::fminimum_magbf16(bfloat16(0.0),
                                                       bfloat16(0.0)));

constexpr bfloat16 totalorderbf16_x = bfloat16(0.0);
constexpr bfloat16 totalorderbf16_y = bfloat16(0.0);
static_assert(1 == LIBC_NAMESPACE::shared::totalorderbf16(&totalorderbf16_x,
                                                          &totalorderbf16_y));
constexpr bfloat16 totalordermagbf16_x = bfloat16(0.0);
constexpr bfloat16 totalordermagbf16_y = bfloat16(0.0);
static_assert(1 ==
              LIBC_NAMESPACE::shared::totalordermagbf16(&totalordermagbf16_x,
                                                        &totalordermagbf16_y));

TEST(LlvmLibcSharedMathTest, ConstantEvaluation) {}
