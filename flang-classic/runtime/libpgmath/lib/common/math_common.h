
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if !(defined __MATH_COMMON_H_INCLUDED__)
#define __MATH_COMMON_H_INCLUDED__ 1

#include <math.h>    // needed for fma declaration
#include <complex.h> // needed for I declaration

#if !(defined INLINE)
#define INLINE __attribute__((always_inline)) inline
#endif

#define GLUE(a,b) a ## b
#define JOIN(a,b) GLUE(a,b)
#define JOIN3(a,b,c)     JOIN(JOIN(a,b),c)
#define JOIN4(a,b,c,d)   JOIN(JOIN3(a,b,c),d)

#include "debug_prn.h"
#include <assert.h>

#define FL_ABS_MASK  0x7fffffff
#define FL_EXP_MASK  0x7f800000
#define FL_PINF      FL_EXP_MASK
#define FL_NINF      0xff800000
#define FL_EXP_BIAS  127
#define FL_EXP_MIN   -149 // corresponds to minimum denormal
#define FL_EXP_MAX   FL_EXP_BIAS
#define FL_SIGN_BIT  0x80000000
#define FL_PREC_BITS 24 // including leading implicit bit
#define FL_ONE       0x3f800000
#define FL_MONE      0xbf800000
#define FL_PZERO     0x0
#define FL_NZERO     FL_SIGN_BIT

#define DB_ABS_MASK  0x7fffffffffffffffULL
#define DB_EXP_MASK  0x7ff0000000000000ULL
#define DB_PINF      DB_EXP_MASK
#define DB_NINF      0xfff0000000000000ULL
#define DB_EXP_BIAS  1023
#define DB_EXP_MIN   -1074 // corresponds to minimum denormal
#define DB_EXP_MAX   DB_EXP_BIAS
#define DB_SIGN_BIT  0x8000000000000000ULL
#define DB_PREC_BITS 53 // including leading implicit bit
#define DB_ONE       0x3ff0000000000000ULL
#define DB_MONE      0xBff0000000000000ULL
#define DB_PZERO     0x0ULL
#define DB_NZERO     DB_SIGN_BIT

static INLINE
float _Complex set_cmplx(float x, float y)
{
    float _Complex result;
#if defined __INTEL_COMPILER
    result = (x + I*y); PRINT(result); // this causes GCC to emit multiply by zero, because I is read as complex and not just imaginary
#else
    *(0 + (float *)(&result)) = x;
    *(1 + (float *)(&result)) = y;
#endif
                                        PRINT(result);
    return result;
}

static INLINE
double _Complex set_cmplxd(double x, double y)
{
    double _Complex result;
    *(0 + (double *)(&result)) = x;
    *(1 + (double *)(&result)) = y;
                                        PRINT(result);
    return result;
}

static INLINE
double set_dcmplx(float x, float y)
{
    double result;
    *(0 + (float *)(&result)) = x;
    *(1 + (float *)(&result)) = y;
                                        PRINT(result);
    return result;
}

#if (defined __INTEL_COMPILER)
#include <immintrin.h>
#endif

#undef fmaf
#define fmaf my_fmaf
static INLINE
float my_fmaf(float a, float b, float c)
{
#if (defined __INTEL_COMPILER)
    // ICC somehow doesn't recognize fmaf as builtin under fp-model source
    // this hack improves performance as FMA function call is slow, but it
    // also breaks vectorization because of the use of mm types.
    #define F2MM(x) _mm_set_ss(x)
    return _mm_cvtss_f32(_mm_fmadd_ss(F2MM(a), F2MM(b), F2MM(c)));
#elif (defined __PGI)
    // PGI also doesn't recognize the function as builtin, yet it doesn't
    // support Intel intrinsics, so below is just to allow inlining of the
    // function.
    // FIXME: it breaks FP program
    return a*b + c;
#else
    // same thing happened with older GCC/clang
    return __builtin_fmaf(a, b, c);
#endif
}

#if (defined __clang__) || (defined __GNUC__)
#undef fma
#define fma my_fma
static INLINE
double my_fma(double a, double b, double c)
{
    // fma() allegedly wasn't recognized by some older GCC and/or clang
    return __builtin_fma(a, b, c);
}
#endif

static INLINE
unsigned F2I(float x)
{
#if defined __INTEL_COMPILER
    return _castf32_u32(x);
#else
    return (*(unsigned *)(&(x)));
#endif
}

static INLINE
float    I2F(unsigned x)
{
#if defined __INTEL_COMPILER
    return _castu32_f32(x);
#else
    return (*(float *)(&(x)));
#endif
}

static INLINE
unsigned long long int D2L(double x)
{
#if defined __INTEL_COMPILER
    return _castf64_u64(x);
#else
    return (*(unsigned long long *)(&(x)));
#endif
}
static INLINE
double L2D(unsigned long long int x)
{
#if defined __INTEL_COMPILER
    return _castu64_f64(x);
#else
    return (*(double *)(&(x)));
#endif
}

static INLINE
unsigned long long int II2L(unsigned hi, unsigned lo)
{
    return (unsigned long long int)lo | (((unsigned long long int)hi) << 32);
}

#undef isinff
#define isinff(x) my_isinff(x)
static INLINE
int my_isinff(float x)
{
    return ((F2I(x) & 0x7fffffff) == 0x7f800000);
}

#undef copysignf
#define copysignf(x, y) my_copysignf(x, y)
static INLINE
float my_copysignf(float x, float y)
{
    return I2F( (F2I(x) & FL_ABS_MASK) | (F2I(y) & FL_SIGN_BIT) );
}

#undef copysign
#define copysign(x, y) my_copysign(x, y)
static INLINE
double my_copysign(double x, double y)
{
    return L2D( (D2L(x) & DB_ABS_MASK) | (D2L(y) & DB_SIGN_BIT) );
}

#undef isnanf
#define isnanf(x) my_isnanf(x)
static INLINE
int my_isnanf(float x)
{
    return ((F2I(x) & 0x7fffffff) > 0x7f800000);
}

static INLINE
void fast2mul(float x, float y, float *r1, float *r2)
{
    float p1 = x*y;
    float p2 = fmaf(x, y, -p1);
    *r1 = p1;
    *r2 = p2;
    return;
}

static INLINE
void fast2sum(float x, float y, float *r1, float *r2)
{
    float hi, tmp, lo;
    hi  = x + y;
    tmp = hi - x;
    lo  = y - tmp;
    *r1 = hi;
    *r2 = lo;
    return;
}

static INLINE
void fast2mul_dp(double x, double y, double *r1, double *r2)
{
    PRINT(x); PRINT(y);
    double fast2mul_p1 = x*y;                                 PRINT(fast2mul_p1);
    double fast2mul_p2 = fma(x, y, -fast2mul_p1);             PRINT(fast2mul_p2);
    *r1 = fast2mul_p1;
    *r2 = fast2mul_p2;
    return;
}

static INLINE
void fast2sum_dp(double x, double y, double *r1, double *r2)
{
    double hi, tmp, lo;
    hi  = x + y;
    tmp = hi - x;
    lo  = y - tmp;
    *r1 = hi;
    *r2 = lo;
    return;
}

#undef creal
#define creal my_creal
static INLINE
double my_creal(double _Complex x)
{
    return *(0 + (double *)&x);
}

#undef cimag
#define cimag my_cimag
static INLINE
double my_cimag(double _Complex x)
{
    return *(1 + (double *)&x);
}

#undef crealf
#define crealf my_crealf
static INLINE
float my_crealf(float _Complex x)
{
    return *(0 + (float *)&x);
}

#undef cimagf
#define cimagf my_cimagf
static INLINE
float my_cimagf(float _Complex x)
{
    return *(1 + (float *)&x);
}

#endif //!(defined __MATH_COMMON_H_INCLUDED__)
