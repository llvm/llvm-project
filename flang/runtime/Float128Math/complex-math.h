/*===-- runtime/Float128Math/complex-math.h -------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/

#ifndef FORTRAN_RUNTIME_FLOAT128MATH_COMPLEX_MATH_H_
#define FORTRAN_RUNTIME_FLOAT128MATH_COMPLEX_MATH_H_

#include "flang/Common/float128.h"
#include "flang/Runtime/entry-names.h"

#if HAS_QUADMATHLIB
#include "quadmath.h"
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
#elif LDBL_MANT_DIG == 113
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
/* We can use __float128 versions of libm functions.
 * __STDC_WANT_IEC_60559_TYPES_EXT__ needs to be defined
 * before including math.h to enable the *f128 prototypes. */
#error "Float128Math build with glibc>=2.26 is unsupported yet"
#endif

#endif /* FORTRAN_RUNTIME_FLOAT128MATH_COMPLEX_MATH_H_ */
