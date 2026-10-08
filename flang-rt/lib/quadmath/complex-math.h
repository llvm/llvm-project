/*===-- lib/quadmath/complex-math.h ---------------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/

#ifndef FLANG_RT_QUADMATH_COMPLEX_MATH_H_
#define FLANG_RT_QUADMATH_COMPLEX_MATH_H_

#include "flang/Common/float128.h"
#include "flang/Runtime/entry-names.h"

#if HAS_QUADMATHLIB
#include "quadmath_wrapper.h"
#define CAbs(x) cabsq(x)
#define CAcos(x) cacosq(x)
#define CAcosh(x) cacoshq(x)
#define CAsin(x) casinq(x)
#define CAsinh(x) casinhq(x)
#define CAtan(x) catanq(x)
#define CAtanh(x) catanhq(x)
#define CCos(x) ccosq(x)
#define CCosh(x) ccoshq(x)
#define CExp(x) cexpq(x)
#define CLog(x) clogq(x)
#define CPow(x, p) cpowq(x, p)
#define CSin(x) csinq(x)
#define CSinh(x) csinhq(x)
#define CSqrt(x) csqrtq(x)
#define CTan(x) ctanq(x)
#define CTanh(x) ctanhq(x)
#elif HAS_LDBL128
/* Use 'long double' versions of libm functions. */
#include <complex.h>

#define CAbs(x) cabsl(x)
#define CAcos(x) cacosl(x)
#define CAcosh(x) cacoshl(x)
#define CAsin(x) casinl(x)
#define CAsinh(x) casinhl(x)
#define CAtan(x) catanl(x)
#define CAtanh(x) catanhl(x)
#define CCos(x) ccosl(x)
#define CCosh(x) ccoshl(x)
#define CExp(x) cexpl(x)
#define CLog(x) clogl(x)
#define CPow(x, p) cpowl(x, p)
#define CSin(x) csinl(x)
#define CSinh(x) csinhl(x)
#define CSqrt(x) csqrtl(x)
#define CTan(x) ctanl(x)
#define CTanh(x) ctanhl(x)
#elif HAS_LIBMF128
/* glibc 2.26 and later export the complex *f128 entry points from libm, so
 * this route needs no third-party library.
 *
 * __STDC_WANT_IEC_60559_TYPES_EXT__ must be defined before <complex.h> and is
 * set on the target in this directory's CMakeLists.txt, not here: a header
 * cannot guarantee it is included first, and the failure is silent. Unlike the
 * scalar half, <complex.h> on glibc 2.43 really does hide cabsf128 without the
 * macro, so this is the half that would break.
 *
 * The argument type is CFloat128ComplexType from flang/Common/float128.h,
 * which is _Complex float __attribute__((mode(TC))). That is what the
 * libquadmath branch already passes to the *q functions, and it is ABI
 * compatible with the _Float128 _Complex these take: verified with the build
 * clang under -Wall, sizeof 32, cabs(2+i) correct to nine places. The C23
 * spelling _Complex _Float128 is not accepted by this clang, and the GNU
 * spelling __complex__ _Float128 is accepted only with a warning that it is
 * being read as _Complex double - which would silently halve the precision. */
#include <complex.h>

/* Compile-time canary: see the note in math-entries.h. If the prototypes are
 * not visible, this fails here rather than at link time. */
__attribute__((unused)) static CFloat128Type (*const c_f128_prototype_canary)(
    CFloat128ComplexType) = &cabsf128;

#define CAbs(x) cabsf128(x)
#define CAcos(x) cacosf128(x)
#define CAcosh(x) cacoshf128(x)
#define CAsin(x) casinf128(x)
#define CAsinh(x) casinhf128(x)
#define CAtan(x) catanf128(x)
#define CAtanh(x) catanhf128(x)
#define CCos(x) ccosf128(x)
#define CCosh(x) ccoshf128(x)
#define CExp(x) cexpf128(x)
#define CLog(x) clogf128(x)
#define CSin(x) csinf128(x)
#define CSinh(x) csinhf128(x)
#define CSqrt(x) csqrtf128(x)
#define CTan(x) ctanf128(x)
#define CTanh(x) ctanhf128(x)
#define CPow(x, p) cpowf128(x, p)
#endif

#endif /* FLANG_RT_QUADMATH_COMPLEX_MATH_H_ */
