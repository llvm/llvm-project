/* ============================================================
Copyright (c) 2002-2015 Advanced Micro Devices, Inc.

All rights reserved.

Redistribution and  use in source and binary  forms, with or
without  modification,  are   permitted  provided  that  the
following conditions are met:

+ Redistributions  of source  code  must  retain  the  above
  copyright  notice,   this  list  of   conditions  and  the
  following disclaimer.

+ Redistributions  in binary  form must reproduce  the above
  copyright  notice,   this  list  of   conditions  and  the
  following  disclaimer in  the  documentation and/or  other
  materials provided with the distribution.

+ Neither the  name of Advanced Micro Devices,  Inc. nor the
  names  of  its contributors  may  be  used  to endorse  or
  promote  products  derived   from  this  software  without
  specific prior written permission.

THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
(INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
(INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
POSSIBILITY OF SUCH DAMAGE.

It is  licensee's responsibility  to comply with  any export
regulations applicable in licensee's jurisdiction.
============================================================ */
#ifndef LIBM_AMD_H_INCLUDED
#define LIBM_AMD_H_INCLUDED 1

/* The following definition of weak_alias is extracted from
   libc-symbols.h */

/* Define ALIASNAME as a weak alias for NAME.
   If weak aliases are not available, this defines a strong alias.  */
#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname)                                           \
  extern __typeof(name) aliasname __attribute__((weak, alias(#name)));

/*
 * We prepend function names by a double underscore.
 *
 *
 * Also allow for a suffix to be specified for all function names that will
 * be included with the function macro FN_PROTOTYPE()
 */

#define _concat3(l,m,r) l##m##r
#define concat3(l,m,r) _concat3(l,m,r)
#if	! defined(FN_PROTO_SUFFIX)
#define	FN_PROTO_SUFFIX
#endif
#define FN_PROTOTYPE(fname) concat3(__, fname,FN_PROTO_SUFFIX)

#include <math.h>

#if !defined(TARGET_WIN)
typedef long __INT8_T;
typedef unsigned long __UINT8_T;

#else
/*****  hacks for windows & open tools *****/
typedef long long __INT8_T;
typedef unsigned long long __UINT8_T;

/* Open Tools #defines ldexpf */
#undef ldexpf


/* Types of exceptions in the `type' field.  */
#if	! defined(TARGET_WIN)
# define DOMAIN         1
# define SING           2
# define OVERFLOW       3
# define UNDERFLOW      4
# define TLOSS          5
# define PLOSS          6
#endif

#endif

#ifndef _COMPLEX_DEFINED
struct _complex {
  double x, y; /* real and imaginary parts */
};
#define _COMPLEX_DEFINED
#endif

#define COMPLEX struct _complex

extern double FN_PROTOTYPE(mth_i_dexp2)(double x);
extern double FN_PROTOTYPE(mth_i_dlog2)(double x);
#if	! defined(TARGET_WIN)
extern double acos(double x);
extern float acosf(float x);

extern double asin(double x);
extern float asinf(float x);

extern double atan(double x);
extern float atanf(float x);

extern double atan2(double x, double y);
extern float atan2f(float x, float y);

extern double ceil(double x);
extern float ceilf(float x);

extern double chgsign(double x);
extern float chgsignf(float x);

extern double copysign(double x, double y);
extern float copysignf(float x, float y);

extern double cos(double x);
extern float cosf(float x);

extern double cosh(double x);
extern float coshf(float x);

extern double exp(double x);
extern float expf(float x);

extern float exp2f(float x);

extern double exp10(double x);
extern float exp10f(float x);

extern double fdim(double x, double y);
extern float fdimf(float x, float y);

extern int finite(double x);
extern int finitef(float x);

extern double floor(double x);
extern float floorf(float x);

extern double fma(double x, double y, double z);
extern float fmaf(float x, float y, float z);

extern double fmax(double x, double y);
extern float fmaxf(float x, float y);

extern double fmin(double x, double y);
extern float fminf(float x, float y);

extern double fmod(double x, double y);
extern float fmodf(float x, float y);

extern double hypot(double x, double y);
extern float hypotf(float x, float y);

extern double ldexp(double x, int exp);
extern float ldexpf(float x, int exp);

extern double log(double x);
extern float logf(float x);

extern float log2f(float x);

extern double log10(double x);
extern float log10f(float x);

extern double logb(double x);
extern float logbf(float x);

extern double modf(double x, double *iptr);
extern float modff(float x, float *iptr);

extern double nextafter(double x, double y);
extern float nextafterf(float x, float y);

extern double pow(double x, double y);
extern float powf(float x, float y);

extern double pow10(double x);
extern float pow10f(float x);

extern double remainder(double x, double y);
extern float remainderf(float x, float y);

extern void __remainder_piby2(double x, double *r, double *rr, int *region);
extern void __remainder_piby2f(float x, double *r, int *region);

extern double sin(double x);
extern float sinf(float x);

extern void sincos(double x, double *s, double *c);
extern void sincosf(float x, float *s, float *c);

extern double sinh(double x);
extern float sinhf(float x);

extern double sqrt(double x);
extern float sqrtf(float x);

extern double tan(double x);
extern float tanf(float x);

extern double tanh(double x);
extern float tanhf(float x);

extern double trunc(double x);
extern float truncf(float x);
#endif

#endif /* LIBM_AMD_H_INCLUDED */
