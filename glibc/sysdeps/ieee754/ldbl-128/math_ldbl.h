/* Manipulation of the bit representation of 'long double' quantities.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_LDBL_H_
#define _MATH_LDBL_H_ 1

#include <stdint.h>
#include <endian.h>

/* A union which permits us to convert between a long double and
   four 32 bit ints or two 64 bit ints.  */

#if __FLOAT_WORD_ORDER == __BIG_ENDIAN

typedef union
{
  long double value;
  struct
  {
    uint64_t msw;
    uint64_t lsw;
  } parts64;
  struct
  {
    uint32_t w0, w1, w2, w3;
  } parts32;
} ieee854_long_double_shape_type;

#endif

#if __FLOAT_WORD_ORDER == __LITTLE_ENDIAN

typedef union
{
  long double value;
  struct
  {
    uint64_t lsw;
    uint64_t msw;
  } parts64;
  struct
  {
    uint32_t w3, w2, w1, w0;
  } parts32;
} ieee854_long_double_shape_type;

#endif

/* Get two 64 bit ints from a long double.  */

#define GET_LDOUBLE_WORDS64(ix0,ix1,d)				\
do {								\
  ieee854_long_double_shape_type qw_u;				\
  qw_u.value = (d);						\
  (ix0) = qw_u.parts64.msw;					\
  (ix1) = qw_u.parts64.lsw;					\
} while (0)

/* Set a long double from two 64 bit ints.  */

#define SET_LDOUBLE_WORDS64(d,ix0,ix1)				\
do {								\
  ieee854_long_double_shape_type qw_u;				\
  qw_u.parts64.msw = (ix0);					\
  qw_u.parts64.lsw = (ix1);					\
  (d) = qw_u.value;						\
} while (0)

/* Get the more significant 64 bits of a long double mantissa.  */

#define GET_LDOUBLE_MSW64(v,d)					\
do {								\
  ieee854_long_double_shape_type sh_u;				\
  sh_u.value = (d);						\
  (v) = sh_u.parts64.msw;					\
} while (0)

/* Set the more significant 64 bits of a long double mantissa from an int.  */

#define SET_LDOUBLE_MSW64(d,v)					\
do {								\
  ieee854_long_double_shape_type sh_u;				\
  sh_u.value = (d);						\
  sh_u.parts64.msw = (v);					\
  (d) = sh_u.value;						\
} while (0)

/* Get the least significant 64 bits of a long double mantissa.  */

#define GET_LDOUBLE_LSW64(v,d)					\
do {								\
  ieee854_long_double_shape_type sh_u;				\
  sh_u.value = (d);						\
  (v) = sh_u.parts64.lsw;					\
} while (0)

/*
   On a platform already supporting a binary128 long double,
   _Float128 will alias to long double.  This transformation
   makes aliasing *l functions to *f128 trivial.
*/
#define _Float128 long double
#define L(x) x##L

#endif /* math_ldbl.h */
