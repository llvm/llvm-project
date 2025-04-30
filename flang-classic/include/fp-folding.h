/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief API for floating-point folding
 *
 *  Define an API for compile-time evaluation of floating-point
 *  operations and math library functions.
 *
 *  The implementation use the host's floating-point arithmetic
 *  and libraries, but it could also implement these interfaces
 *  with software emulation, such as by using Berkeley SoftFloat.
 *
 *  Note: All of these interfaces receive their actual arguments as
 *  pointers to constant floating-point types.  This protocol is
 *  less convenient to use, due to its need for explicit intermediate
 *  temporary variables, but it reduces the risk of a silent
 *  numeric conversion of an argument value.
 *
 *  Note: The types have names like "float32_t" to distinguish them
 *  from integers, while the routines use names like "real32" to
 *  distinguish them from complex operations that might be added.
 */

#ifndef FP_FOLDING_H_
#define FP_FOLDING_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>
#include <float.h>
#include <stdint.h>

/*
 *  At most one of these FOLD_LDBL_... macros will be defined here
 *  to identify the native "long double" type.
 */
#if DECIMAL_DIG == 21 && LDBL_MANT_DIG == 64
# define FOLD_LDBL_X87 1 /* x87 FPU 80-bit extended precision */
#elif DECIMAL_DIG == 33 && LDBL_MANT_DIG == 106
# define FOLD_LDBL_DOUBLEDOUBLE 1 /* Power(PC) double-double */
#elif DECIMAL_DIG > 33 && LDBL_MANT_DIG == 112
# define FOLD_LDBL_128BIT 1 /* 128-bit IEEE */
#elif DECIMAL_DIG == 36 && LDBL_MANT_DIG == 113
# define FOLD_LDBL_128BIT 1 /* 128-bit IEEE */
#elif DECIMAL_DIG == 17 && LDBL_MANT_DIG == 53
# define FOLD_LDBL_JUST_DOUBLE /* long double is a synonym for IEEE double */
#else
# warning unrecognized long double configuration (DECIMAL_DIG, LDBL_MANT_DIG)
#endif

typedef float float32_t;
typedef double float64_t;
typedef long double float128_t; /* 128 bits in memory, format host-dependent */

#if !defined(_WIN32)
# define FLOAT_COMPLEX_TYPE float complex
# define FLOAT_COMPLEX_CREATE(real, imag) (real + imag * I)
# define DOUBLE_COMPLEX_TYPE double complex
# define DOUBLE_COMPLEX_CREATE(real, imag) (real + imag * I)
# define LONG_DOUBLE_COMPLEX_TYPE long double complex
# define LONG_DOUBLE_COMPLEX_CREATE(real, imag) (real + imag * I)

/* For type conversion */
# define FCMPLX_TO_DCMPLX(cplx) ((double complex)cplx)
# define DCMPLX_TO_FCMPLX(cplx) ((float complex)cplx)
# define DCMPLX_TO_LCMPLX(cplx) ((long double complex)cplx)
# define LCMPLX_TO_DCMPLX(cplx) ((double complex)cplx)
#else
# define FLOAT_COMPLEX_TYPE _Fcomplex
# define FLOAT_COMPLEX_CREATE(real, imag) _FCbuild(real, imag)
# define DOUBLE_COMPLEX_TYPE _Dcomplex
# define DOUBLE_COMPLEX_CREATE(real, imag) _Cbuild(real, imag)
# define LONG_DOUBLE_COMPLEX_TYPE _Lcomplex
# define LONG_DOUBLE_COMPLEX_CREATE(real, imag) _LCbuild(real, imag)

/*
 * For type conversion
 * On Windows the complex numbers aren't native type,
 * so the standard arithmetic operators aren't defined on complex types.
 * Implicit conversions aren't defined between complex types.
 */
# define FCMPLX_TO_DCMPLX(cplx) DOUBLE_COMPLEX_CREATE((double)crealf(cplx), (double)cimagf(cplx))
# define DCMPLX_TO_FCMPLX(cplx) FLOAT_COMPLEX_CREATE((float)creal(cplx), (float)cimag(cplx))
# define DCMPLX_TO_LCMPLX(cplx) LONG_DOUBLE_COMPLEX_CREATE((long double)creal(cplx), (long double)cimag(cplx))
# define LCMPLX_TO_DCMPLX(cplx) DOUBLE_COMPLEX_CREATE((double)creall(cplx), (double)cimagl(cplx))
#endif // !defined(_WIN64)

void fold_sanity_check(void);

/*
 *  Comparisons.  These can't trap.
 */

enum fold_relation {
  FOLD_LT = -1,
  FOLD_EQ = 0,
  FOLD_GT = 1,
  FOLD_UN = -2,
};

enum fold_relation fold_real32_compare(const float32_t *x, const float32_t *y);
enum fold_relation fold_real64_compare(const float64_t *x, const float64_t *y);
enum fold_relation fold_real128_compare(const float128_t *x, const float128_t *y);

/*
 *  Operations.  These all return a status code.
 */

enum fold_status {
  FOLD_OK = 0,
  FOLD_INVALID = -1,
  FOLD_OVERFLOW = -2, /* infinite result, including DIVBYZERO */
  FOLD_UNDERFLOW = -3, /* result flushed to zero */
  FOLD_INEXACT = -4,
};

enum fold_status fold_int32_from_real32(int32_t *res, const float32_t *arg);
enum fold_status fold_int64_from_real32(int64_t *res, const float32_t *arg);
enum fold_status fold_uint64_from_real32(uint64_t *res, const float32_t *arg);
enum fold_status fold_uint32_from_real32(uint32_t *res, const float32_t *arg);
enum fold_status fold_real32_from_int64(float32_t *res, const int64_t *arg);
enum fold_status fold_real32_from_uint64(float32_t *res, const uint64_t *arg);
enum fold_status fold_real32_from_real64(float32_t *res, const float64_t *arg);
enum fold_status fold_real32_from_real128(float32_t *res, const float128_t *arg);
enum fold_status fold_real32_negate(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_abs(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_sqrt(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_add(float32_t *res, const float32_t *x, const float32_t *y);
enum fold_status fold_real32_subtract(float32_t *res, const float32_t *x, const float32_t *y);
enum fold_status fold_real32_multiply(float32_t *res, const float32_t *x, const float32_t *y);
enum fold_status fold_real32_divide(float32_t *res, const float32_t *x, const float32_t *y);
enum fold_status fold_real32_pow(float32_t *res, const float32_t *x, const float32_t *y);
enum fold_status fold_real32_sin(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_cos(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_tan(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_asin(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_acos(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_atan(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_atan2(float32_t *res, const float32_t *x, const float32_t *y);
enum fold_status fold_real32_exp(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_log(float32_t *res, const float32_t *arg);
enum fold_status fold_real32_log10(float32_t *res, const float32_t *arg);
enum fold_status fold_int32_from_real64(int32_t *res, const float64_t *arg);
enum fold_status fold_int64_from_real64(int64_t *res, const float64_t *arg);
enum fold_status fold_uint32_from_real64(uint32_t *res, const float64_t *arg);
enum fold_status fold_uint64_from_real64(uint64_t *res, const float64_t *arg);
enum fold_status fold_real64_from_int64(float64_t *res, const int64_t *arg);
enum fold_status fold_real64_from_uint64(float64_t *res, const uint64_t *arg);
enum fold_status fold_real64_from_real32(float64_t *res, const float32_t *arg);
enum fold_status fold_real64_from_real128(float64_t *res, const float128_t *arg);
enum fold_status fold_real64_negate(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_abs(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_sqrt(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_add(float64_t *res, const float64_t *x, const float64_t *y);
enum fold_status fold_real64_subtract(float64_t *res, const float64_t *x, const float64_t *y);
enum fold_status fold_real64_multiply(float64_t *res, const float64_t *x, const float64_t *y);
enum fold_status fold_real64_divide(float64_t *res, const float64_t *x, const float64_t *y);
enum fold_status fold_real64_pow(float64_t *res, const float64_t *x, const float64_t *y);
enum fold_status fold_real64_sin(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_cos(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_tan(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_asin(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_acos(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_atan(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_atan2(float64_t *res, const float64_t *x, const float64_t *y);
enum fold_status fold_real64_exp(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_log(float64_t *res, const float64_t *arg);
enum fold_status fold_real64_log10(float64_t *res, const float64_t *arg);
enum fold_status fold_int32_from_real128(int32_t *res, const float128_t *arg);
enum fold_status fold_int64_from_real128(int64_t *res, const float128_t *arg);
enum fold_status fold_uint32_from_real128(uint32_t *res, const float128_t *arg);
enum fold_status fold_uint64_from_real128(uint64_t *res, const float128_t *arg);
enum fold_status fold_real128_from_int64(float128_t *res, const int64_t *arg);
enum fold_status fold_real128_from_uint64(float128_t *res, const uint64_t *arg);
enum fold_status fold_real128_from_real32(float128_t *res, const float32_t *arg);
enum fold_status fold_real128_from_real64(float128_t *res, const float64_t *arg);
enum fold_status fold_real128_negate(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_abs(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_sqrt(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_add(float128_t *res, const float128_t *x, const float128_t *y);
enum fold_status fold_real128_subtract(float128_t *res, const float128_t *x, const float128_t *y);
enum fold_status fold_real128_multiply(float128_t *res, const float128_t *x, const float128_t *y);
enum fold_status fold_real128_divide(float128_t *res, const float128_t *x, const float128_t *y);
enum fold_status fold_real128_pow(float128_t *res, const float128_t *x, const float128_t *y);
enum fold_status fold_real128_sin(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_cos(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_tan(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_asin(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_acos(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_atan(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_atan2(float128_t *res, const float128_t *x, const float128_t *y);
enum fold_status fold_real128_exp(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_log(float128_t *res, const float128_t *arg);
enum fold_status fold_real128_log10(float128_t *res, const float128_t *arg);

enum fold_status fold_complex32_pow(FLOAT_COMPLEX_TYPE *res, const FLOAT_COMPLEX_TYPE *x,
                                    const FLOAT_COMPLEX_TYPE *y);
enum fold_status fold_complex64_pow(DOUBLE_COMPLEX_TYPE *res,
                                    const DOUBLE_COMPLEX_TYPE *x,
                                    const DOUBLE_COMPLEX_TYPE *y);
enum fold_status fold_complex128_pow(LONG_DOUBLE_COMPLEX_TYPE *res,
                                     const LONG_DOUBLE_COMPLEX_TYPE *x,
                                     const LONG_DOUBLE_COMPLEX_TYPE *y);

#ifdef __cplusplus
}
#endif
#endif /* FP_FOLDING_H_ */
