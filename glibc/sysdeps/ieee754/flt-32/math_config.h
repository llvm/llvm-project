/* Configuration for math routines.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

static inline uint32_t
asuint (float f)
{
  union
  {
    float f;
    uint32_t i;
  } u = {f};
  return u.i;
}

static inline float
asfloat (uint32_t i)
{
  union
  {
    uint32_t i;
    float f;
  } u = {i};
  return u.f;
}

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

#define NOINLINE __attribute__ ((noinline))

attribute_hidden float __math_oflowf (uint32_t);
attribute_hidden float __math_uflowf (uint32_t);
attribute_hidden float __math_may_uflowf (uint32_t);
attribute_hidden float __math_divzerof (uint32_t);
attribute_hidden float __math_invalidf (float);

/* Shared between expf, exp2f, exp10f, and powf.  */
#define EXP2F_TABLE_BITS 5
#define EXP2F_POLY_ORDER 3
extern const struct exp2f_data
{
  uint64_t tab[1 << EXP2F_TABLE_BITS];
  double shift_scaled;
  double poly[EXP2F_POLY_ORDER];
  double shift;
  double invln2_scaled;
  double poly_scaled[EXP2F_POLY_ORDER];
} __exp2f_data attribute_hidden;

#define LOGF_TABLE_BITS 4
#define LOGF_POLY_ORDER 4
extern const struct logf_data
{
  struct
  {
    double invc, logc;
  } tab[1 << LOGF_TABLE_BITS];
  double ln2;
  double poly[LOGF_POLY_ORDER - 1]; /* First order coefficient is 1.  */
} __logf_data attribute_hidden;

#define LOG2F_TABLE_BITS 4
#define LOG2F_POLY_ORDER 4
extern const struct log2f_data
{
  struct
  {
    double invc, logc;
  } tab[1 << LOG2F_TABLE_BITS];
  double poly[LOG2F_POLY_ORDER];
} __log2f_data attribute_hidden;

#define POWF_LOG2_TABLE_BITS 4
#define POWF_LOG2_POLY_ORDER 5
#if TOINT_INTRINSICS
# define POWF_SCALE_BITS EXP2F_TABLE_BITS
#else
# define POWF_SCALE_BITS 0
#endif
#define POWF_SCALE ((double) (1 << POWF_SCALE_BITS))
extern const struct powf_log2_data
{
  struct
  {
    double invc, logc;
  } tab[1 << POWF_LOG2_TABLE_BITS];
  double poly[POWF_LOG2_POLY_ORDER];
} __powf_log2_data attribute_hidden;

#endif
