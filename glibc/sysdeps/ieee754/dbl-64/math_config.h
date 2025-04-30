/* Configuration for double precision math routines.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _MATH_CONFIG_H
#define _MATH_CONFIG_H

#include <math.h>
#include <math_private.h>
#include <nan-high-order-bit.h>
#include <stdint.h>

#ifndef WANT_ROUNDING
/* Correct special case results in non-nearest rounding modes.  */
# define WANT_ROUNDING 1
#endif
#ifndef WANT_ERRNO
/* Set errno according to ISO C with (math_errhandling & MATH_ERRNO) != 0.  */
# define WANT_ERRNO 1
#endif
#ifndef WANT_ERRNO_UFLOW
/* Set errno to ERANGE if result underflows to 0 (in all rounding modes).  */
# define WANT_ERRNO_UFLOW (WANT_ROUNDING && WANT_ERRNO)
#endif

#ifndef TOINT_INTRINSICS
/* When set, the roundtoint and converttoint functions are provided with
   the semantics documented below.  */
# define TOINT_INTRINSICS 0
#endif

#if TOINT_INTRINSICS
/* Round x to nearest int in all rounding modes, ties have to be rounded
   consistently with converttoint so the results match.  If the result
   would be outside of [-2^31, 2^31-1] then the semantics is unspecified.  */
static inline double_t
roundtoint (double_t x);

/* Convert x to nearest int in all rounding modes, ties have to be rounded
   consistently with roundtoint.  If the result is not representible in an
   int32_t then the semantics is unspecified.  */
static inline int32_t
converttoint (double_t x);
#endif

static inline uint64_t
asuint64 (double f)
{
  union
  {
    double f;
    uint64_t i;
  } u = {f};
  return u.i;
}

static inline double
asdouble (uint64_t i)
{
  union
  {
    uint64_t i;
    double f;
  } u = {i};
  return u.f;
}

static inline int
issignaling_inline (double x)
{
  uint64_t ix = asuint64 (x);
  if (HIGH_ORDER_BIT_IS_SET_FOR_SNAN)
    return (ix & 0x7ff8000000000000) == 0x7ff8000000000000;
  return 2 * (ix ^ 0x0008000000000000) > 2 * 0x7ff8000000000000ULL;
}

#define NOINLINE __attribute__ ((noinline))

/* Error handling tail calls for special cases, with a sign argument.
   The sign of the return value is set if the argument is non-zero.  */

/* The result overflows.  */
attribute_hidden double __math_oflow (uint32_t);
/* The result underflows to 0 in nearest rounding mode.  */
attribute_hidden double __math_uflow (uint32_t);
/* The result underflows to 0 in some directed rounding mode only.  */
attribute_hidden double __math_may_uflow (uint32_t);
/* Division by zero.  */
attribute_hidden double __math_divzero (uint32_t);

/* Error handling using input checking.  */

/* Invalid input unless it is a quiet NaN.  */
attribute_hidden double __math_invalid (double);

/* Error handling using output checking, only for errno setting.  */

/* Check if the result overflowed to infinity.  */
attribute_hidden double __math_check_oflow (double);
/* Check if the result underflowed to 0.  */
attribute_hidden double __math_check_uflow (double);

/* Check if the result overflowed to infinity.  */
static inline double
check_oflow (double x)
{
  return WANT_ERRNO ? __math_check_oflow (x) : x;
}

/* Check if the result underflowed to 0.  */
static inline double
check_uflow (double x)
{
  return WANT_ERRNO ? __math_check_uflow (x) : x;
}

#define EXP_TABLE_BITS 7
#define EXP_POLY_ORDER 5
#define EXP2_POLY_ORDER 5
extern const struct exp_data
{
  double invln2N;
  double shift;
  double negln2hiN;
  double negln2loN;
  double poly[4]; /* Last four coefficients.  */
  double exp2_shift;
  double exp2_poly[EXP2_POLY_ORDER];
  uint64_t tab[2*(1 << EXP_TABLE_BITS)];
} __exp_data attribute_hidden;

#define LOG_TABLE_BITS 7
#define LOG_POLY_ORDER 6
#define LOG_POLY1_ORDER 12
extern const struct log_data
{
  double ln2hi;
  double ln2lo;
  double poly[LOG_POLY_ORDER - 1]; /* First coefficient is 1.  */
  double poly1[LOG_POLY1_ORDER - 1];
  /* See e_log_data.c for details.  */
  struct {double invc, logc;} tab[1 << LOG_TABLE_BITS];
#ifndef __FP_FAST_FMA
  struct {double chi, clo;} tab2[1 << LOG_TABLE_BITS];
#endif
} __log_data attribute_hidden;

#define LOG2_TABLE_BITS 6
#define LOG2_POLY_ORDER 7
#define LOG2_POLY1_ORDER 11
extern const struct log2_data
{
  double invln2hi;
  double invln2lo;
  double poly[LOG2_POLY_ORDER - 1];
  double poly1[LOG2_POLY1_ORDER - 1];
  /* See e_log2_data.c for details.  */
  struct {double invc, logc;} tab[1 << LOG2_TABLE_BITS];
#ifndef __FP_FAST_FMA
  struct {double chi, clo;} tab2[1 << LOG2_TABLE_BITS];
#endif
} __log2_data attribute_hidden;

#define POW_LOG_TABLE_BITS 7
#define POW_LOG_POLY_ORDER 8
extern const struct pow_log_data
{
  double ln2hi;
  double ln2lo;
  double poly[POW_LOG_POLY_ORDER - 1]; /* First coefficient is 1.  */
  /* Note: the pad field is unused, but allows slightly faster indexing.  */
  /* See e_pow_log_data.c for details.  */
  struct {double invc, pad, logc, logctail;} tab[1 << POW_LOG_TABLE_BITS];
} __pow_log_data attribute_hidden;

#endif
