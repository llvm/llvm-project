/* file: libm_cpu_defs.h */


// Copyright (c) 2000 - 2004, Intel Corporation
// All rights reserved.
//
// Contributed 2000 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//

#ifndef __LIBM_CPU_DEFS__H_INCLUDED__
#define __LIBM_CPU_DEFS__H_INCLUDED__

void __libm_sincos_pi4(double,double*,double*,int);
void __libm_y0y1(double , double *, double *);
void __libm_j0j1(double , double *, double *);
double __libm_j0(double);
double __libm_j1(double);
double __libm_jn(int,double);
double __libm_y0(double);
double __libm_y1(double);
double __libm_yn(int,double);

double __libm_copysign (double, double);
float __libm_copysignf (float, float);
long double __libm_copysignl (long double, long double);

extern double sqrt(double);
extern double fabs(double);
extern double log(double);
extern double log1p(double);
extern double sqrt(double);
extern double sin(double);
extern double exp(double);
extern double modf(double, double *);
extern double asinh(double);
extern double acosh(double);
extern double atanh(double);
extern double tanh(double);
extern double erf(double);
extern double erfc(double);
extern double j0(double);
extern double j1(double);
extern double jn(int, double);
extern double y0(double);
extern double y1(double);
extern double yn(int, double);

extern float  fabsf(float);
extern float  asinhf(float);
extern float  acoshf(float);
extern float  atanhf(float);
extern float  tanhf(float);
extern float  erff(float);
extern float  erfcf(float);
extern float  j0f(float);
extern float  j1f(float);
extern float  jnf(int, float);
extern float  y0f(float);
extern float  y1f(float);
extern float  ynf(int, float);

extern long double log1pl(long double);
extern long double logl(long double);
extern long double sqrtl(long double);
extern long double expl(long double);
extern long double fabsl(long double);

#if !(defined(SIZE_LONG_INT_32) || defined(SIZE_LONG_INT_64))
#error long int size not established; define SIZE_LONG_INT_32 or SIZE_LONG_INT_64
#endif

#if (defined(SIZE_LONG_INT_32) && defined(SIZE_LONG_INT_64))
#error multiple long int size definitions; define SIZE_LONG_INT_32 or SIZE_LONG_INT_64
#endif

#if !(defined(SIZE_LONG_LONG_INT_32) || defined(SIZE_LONG_LONG_INT_64))
#error long long int size not established; define SIZE_LONG_LONG_INT_32 or SIZE_LONG_LONG_INT_64
#endif

#if (defined(SIZE_LONG_LONG_INT_32) && defined(SIZE_LONG_LONG_INT_64))
#error multiple long long int size definitions; define SIZE_LONG_LONG_INT_32 or SIZE_LONG_LONG_INT_64
#endif

#define HI_SIGNIFICAND_LESS(X, HI) ((X)->hi_significand < 0x ## HI)
#define f64abs(x) ((x) < 0.0 ? -(x) : (x))

#define FP80_DECLARE()
#define FP80_SET()
#define FP80_RESET()

#ifdef _LIBC
# include <math.h>
#else

static const unsigned INF[] = {
    DOUBLE_HEX(7ff00000, 00000000),
    DOUBLE_HEX(fff00000, 00000000)
};

static const double _zeroo = 0.0;
static const double _bigg = 1.0e300;
static const double _ponee = 1.0;
static const double _nonee = -1.0;

#define INVALID    (_zeroo * *((double*)&INF[0]))
#define PINF       *((double*)&INF[0])
#define NINF       -PINF
#define PINF_DZ    (_ponee/_zeroo)
#define X_TLOSS    1.41484755040568800000e+16
#endif

/* Set these appropriately to make thread Safe */
#define ERRNO_RANGE  errno = ERANGE
#define ERRNO_DOMAIN errno = EDOM

#ifndef _LIBC
#if defined(__ICC) || defined(__ICL) || defined(__ECC) || defined(__ECL)
# pragma warning( disable : 68 )	/* #68: integer conversion resulted in a change of sign */
# pragma warning( disable : 186 )	/* #186: pointless comparison of unsigned integer with zero */
# pragma warning( disable : 1572 )	/* #1572: floating-point equality and inequality comparisons are unreliable */
#endif
#endif

#endif    /*__LIBM_CPU_DEFS__H_INCLUDED__*/
