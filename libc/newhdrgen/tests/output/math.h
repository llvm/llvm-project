//===-- C standard library header math.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MATH_H
#define LLVM_LIBC_MATH_H

#include "__llvm-libc-common.h"
#include "llvm-libc-macros/float16-macros.h"
#include "llvm-libc-macros/math-macros.h"
#include "llvm-libc-types/float128.h"

#if !defined(__LIBC_CONST_ATTR) && defined(__cplusplus) && defined(__GNUC__)
#if __has_attribute(const)
#define __LIBC_CONST_ATTR [[gnu::const]]
#endif
#endif
#if !defined(__LIBC_CONST_ATTR) && defined(__GNUC__)
#if __has_attribute(const)
#define __LIBC_CONST_ATTR __attribute__((const))
#endif
#endif
#if !defined(__LIBC_CONST_ATTR)
#define __LIBC_CONST_ATTR
#endif

#include <llvm-libc-types/double_t.h>
#include <llvm-libc-types/float128.h>
#include <llvm-libc-types/float_t.h>

__BEGIN_C_DECLS

float acosf(float) __NOEXCEPT;

float acoshf(float) __NOEXCEPT;

double asin(double) __NOEXCEPT;

float asinf(float) __NOEXCEPT;

float asinhf(float) __NOEXCEPT;

float atan2f(float, float) __NOEXCEPT;

float atanf(float) __NOEXCEPT;

float atanhf(float) __NOEXCEPT;

int canonicalize(double, double) __NOEXCEPT;

int canonicalizef(float, float) __NOEXCEPT;

int canonicalizel(long double, long double) __NOEXCEPT;

double ceil(double) __NOEXCEPT;

float ceilf(float) __NOEXCEPT;

long double ceill(long double) __NOEXCEPT;

double copysign(double, double) __NOEXCEPT;

float copysignf(float, float) __NOEXCEPT;

long double copysignl(long double, long double) __NOEXCEPT;

double cos(double) __NOEXCEPT;

float cosf(float) __NOEXCEPT;

float coshf(float) __NOEXCEPT;

float erff(float) __NOEXCEPT;

double exp(double) __NOEXCEPT;

double exp10(double) __NOEXCEPT;

float exp10f(float) __NOEXCEPT;

double exp2(double) __NOEXCEPT;

float exp2f(float) __NOEXCEPT;

float exp2m1f(float) __NOEXCEPT;

float expf(float) __NOEXCEPT;

double expm1(double) __NOEXCEPT;

float expm1f(float) __NOEXCEPT;

__LIBC_CONST_ATTR double fabs(double) __NOEXCEPT;

float fabsf(float) __NOEXCEPT;

long double fabsl(long double) __NOEXCEPT;

double fdim(double, double) __NOEXCEPT;

float fdimf(float, float) __NOEXCEPT;

long double fdiml(long double, long double) __NOEXCEPT;

double floor(double) __NOEXCEPT;

float floorf(float) __NOEXCEPT;

long double floorl(long double) __NOEXCEPT;

double fma(double, double, double) __NOEXCEPT;

float fmaf(float, float, float) __NOEXCEPT;

double fmax(double, double) __NOEXCEPT;

float fmaxf(float, float) __NOEXCEPT;

double fmaximum(double, double) __NOEXCEPT;

double fmaximum_mag(double, double) __NOEXCEPT;

double fmaximum_mag_num(double, double) __NOEXCEPT;

float fmaximum_mag_numf(float, float) __NOEXCEPT;

long double fmaximum_mag_numl(long double, long double) __NOEXCEPT;

float fmaximum_magf(float, float) __NOEXCEPT;

long double fmaximum_magl(long double, long double) __NOEXCEPT;

double fmaximum_num(double, double) __NOEXCEPT;

float fmaximum_numf(float, float) __NOEXCEPT;

long double fmaximum_numl(long double, long double) __NOEXCEPT;

float fmaximumf(float, float) __NOEXCEPT;

long double fmaximuml(long double, long double) __NOEXCEPT;

long double fmaxl(long double, long double) __NOEXCEPT;

double fmin(double, double) __NOEXCEPT;

float fminf(float, float) __NOEXCEPT;

double fminimum(double, double) __NOEXCEPT;

double fminimum_mag(double, double) __NOEXCEPT;

double fminimum_mag_num(double, double) __NOEXCEPT;

float fminimum_mag_numf(float, float) __NOEXCEPT;

long double fminimum_mag_numl(long double, long double) __NOEXCEPT;

float fminimum_magf(float, float) __NOEXCEPT;

long double fminimum_magl(long double, long double) __NOEXCEPT;

double fminimum_num(double, double) __NOEXCEPT;

float fminimum_numf(float, float) __NOEXCEPT;

float fminimumf(float, float) __NOEXCEPT;

long double fminimuml(long double, long double) __NOEXCEPT;

long double fminl(long double, long double) __NOEXCEPT;

double fmod(double, double) __NOEXCEPT;

float fmodf(float, float) __NOEXCEPT;

long double fmodl(long double, long double) __NOEXCEPT;

float fmul(double, double) __NOEXCEPT;

double frexp(double, int *) __NOEXCEPT;

float frexpf(float, int *) __NOEXCEPT;

long double frexpl(long double, int *) __NOEXCEPT;

double fromfp(double, int, unsigned int) __NOEXCEPT;

float fromfpf(float, int, unsigned int) __NOEXCEPT;

long double fromfpl(long double, int, unsigned int) __NOEXCEPT;

double fromfpx(double, int, unsigned int) __NOEXCEPT;

float fromfpxf(float, int, unsigned int) __NOEXCEPT;

long double fromfpxl(long double, int, unsigned int) __NOEXCEPT;

double hypot(double, double) __NOEXCEPT;

float hypotf(float, float) __NOEXCEPT;

int ilogb(double) __NOEXCEPT;

int ilogbf(float) __NOEXCEPT;

int ilogbl(long double) __NOEXCEPT;

double ldexp(double, int) __NOEXCEPT;

float ldexpf(float, int) __NOEXCEPT;

long double ldexpl(long double, int) __NOEXCEPT;

long llogb(double) __NOEXCEPT;

long llogbf(float) __NOEXCEPT;

long llogbl(long double) __NOEXCEPT;

long long llrint(double) __NOEXCEPT;

long long llrintf(float) __NOEXCEPT;

long long llrintl(long double) __NOEXCEPT;

long long llround(double) __NOEXCEPT;

long long llroundf(float) __NOEXCEPT;

long long llroundl(long double) __NOEXCEPT;

double log(double) __NOEXCEPT;

double log10(double) __NOEXCEPT;

float log10f(float) __NOEXCEPT;

double log1p(double) __NOEXCEPT;

float log1pf(float) __NOEXCEPT;

double log2(double) __NOEXCEPT;

float log2f(float) __NOEXCEPT;

double logb(double) __NOEXCEPT;

float logbf(float) __NOEXCEPT;

long double logbl(long double) __NOEXCEPT;

float logf(float) __NOEXCEPT;

long lrint(double) __NOEXCEPT;

long lrintf(float) __NOEXCEPT;

long lrintl(long double) __NOEXCEPT;

long lround(double) __NOEXCEPT;

long lroundf(float) __NOEXCEPT;

long lroundl(long double) __NOEXCEPT;

double modf(double, double *) __NOEXCEPT;

float modff(float, float *) __NOEXCEPT;

long double modfl(long double, long double *) __NOEXCEPT;

double nan(const char *) __NOEXCEPT;

float nanf(const char *) __NOEXCEPT;

long double nanl(const char *) __NOEXCEPT;

double nearbyint(double) __NOEXCEPT;

float nearbyintf(float) __NOEXCEPT;

long double nearbyintl(long double) __NOEXCEPT;

double nextafter(double, double) __NOEXCEPT;

float nextafterf(float, float) __NOEXCEPT;

long double nextafterl(long double, long double) __NOEXCEPT;

double nextdown(double) __NOEXCEPT;

float nextdownf(float) __NOEXCEPT;

long double nextdownl(long double) __NOEXCEPT;

double nexttoward(double, long double) __NOEXCEPT;

float nexttowardf(float, long double) __NOEXCEPT;

long double nexttowardl(long double, long double) __NOEXCEPT;

double nextup(double) __NOEXCEPT;

float nextupf(float) __NOEXCEPT;

long double nextupl(long double) __NOEXCEPT;

double pow(double, double) __NOEXCEPT;

float powf(float, float) __NOEXCEPT;

double remainder(double, double) __NOEXCEPT;

float remainderf(float, float) __NOEXCEPT;

long double remainderl(long double, long double) __NOEXCEPT;

double remquo(double, double, int *) __NOEXCEPT;

float remquof(float, float, int *) __NOEXCEPT;

long double remquol(long double, long double, int *) __NOEXCEPT;

double rint(double) __NOEXCEPT;

float rintf(float) __NOEXCEPT;

long double rintl(long double) __NOEXCEPT;

double round(double) __NOEXCEPT;

double roundeven(double) __NOEXCEPT;

float roundevenf(float) __NOEXCEPT;

long double roundevenl(long double) __NOEXCEPT;

float roundf(float) __NOEXCEPT;

long double roundl(long double) __NOEXCEPT;

double scalbn(double, int) __NOEXCEPT;

float scalbnf(float, int) __NOEXCEPT;

long double scalbnl(long double, int) __NOEXCEPT;

double sin(double) __NOEXCEPT;

void sincosf(float, float *, float *) __NOEXCEPT;

float sinf(float) __NOEXCEPT;

float sinhf(float) __NOEXCEPT;

double sqrt(double) __NOEXCEPT;

float sqrtf(float) __NOEXCEPT;

long double sqrtl(long double) __NOEXCEPT;

double tan(double) __NOEXCEPT;

float tanf(float) __NOEXCEPT;

float tanhf(float) __NOEXCEPT;

double trunc(double) __NOEXCEPT;

float truncf(float) __NOEXCEPT;

long double truncl(long double) __NOEXCEPT;

double ufromfp(double, int, unsigned int) __NOEXCEPT;

float ufromfpf(float, int, unsigned int) __NOEXCEPT;

long double ufromfpl(long double, int, unsigned int) __NOEXCEPT;

double ufromfpx(double, int, unsigned int) __NOEXCEPT;

float ufromfpxf(float, int, unsigned int) __NOEXCEPT;

long double ufromfpxl(long double, int, unsigned int) __NOEXCEPT;

#ifdef LIBC_TYPES_HAS_FLOAT128
int canonicalizef128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 ceilf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 copysignf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fabsf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fdimf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 floorf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fmaxf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fmaximum_mag_numf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fmaximum_magf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fmaximum_numf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fmaximumf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fminf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fminimum_mag_numf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fminimum_magf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fminimum_numf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fminimumf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fmodf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 frexpf128(float128, int *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fromfpf128(float128, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 fromfpxf128(float128, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
int ilogbf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 ldexpf128(float128, int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
long llogbf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
long long llrintf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
long long llroundf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 logbf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
long lrintf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
long lroundf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 modff128(float128, float128 *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 nanf128(const char *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 nearbyintf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 nextafterf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 nextdownf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 nextupf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 remquof128(float128, float128, int *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 rintf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 roundevenf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 roundf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 scalbnf128(float128, int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 sqrtf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 truncf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 ufromfpf128(float128, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 ufromfpxf128(float128, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT16
int canonicalizef16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 ceilf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 copysignf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16add(double, double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16addf(float, float) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16addl(long double, long double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16div(double, double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16divf(float, float) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16divl(long double, long double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16fma(double, double, double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16fmaf(float, float, float) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16fmal(long double, long double, long double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16sqrt(double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16sqrtf(float) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16sqrtl(long double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16sub(double, double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16subf(float, float) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 f16subl(long double, long double) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fabsf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fdimf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 floorf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fmaxf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fmaximum_mag_numf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fmaximum_magf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fmaximum_numf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fmaximumf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fminf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fminimum_mag_numf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fminimum_magf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fminimum_numf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fminimumf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fmodf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 frexpf16(_Float16, int *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fromfpf16(_Float16, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 fromfpxf16(_Float16, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 getpayloadf16(_Float16 *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
int ilogbf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 ldexpf16(_Float16, int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
long llogbf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
long long llrintf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
long long llroundf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 logbf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
long lrintf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
long lroundf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 modff16(_Float16, _Float16 *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 nanf16(const char *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 nearbyintf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 nextafterf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 nextdownf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 nexttowardf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 nextupf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 remainderf16(_Float16, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 remquof16(_Float16, _Float16, int *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 rintf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 roundevenf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 roundf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 scalblnf16(_Float16, long) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 scalbnf16(_Float16, int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
int setpayloadf16(_Float16 *, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
int setpayloadsigf16(_Float16 *, _Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
int totalorderf16(_Float16 *, _Float16 *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
int totalordermagf16(_Float16 *, _Float16 *) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 truncf16(_Float16) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 ufromfpf16(_Float16, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16
_Float16 ufromfpxf16(_Float16, int, unsigned int) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128
_Float16 f16addf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128
_Float16 f16divf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128
_Float16 f16fmaf128(float128, float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128
_Float16 f16sqrtf128(float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128
_Float16 f16subf128(float128, float128) __NOEXCEPT;
#endif // LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128

__END_C_DECLS

#undef __LIBC_CONST_ATTR

#endif // LLVM_LIBC_MATH_H
