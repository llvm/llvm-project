/* _Float128 IEEE like macros.
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
#ifndef _IEEE754_FLOAT128_H
#define _IEEE754_FLOAT128_H

#include <endian.h>
#include <stdint.h>

# if __FLOAT_WORD_ORDER == BIG_ENDIAN
#  define __FLT_EORDER2(t, a, b) t a; t b;
#  define __FLT_EORDER4(t, a, b, c, d) \
			t a; t b; t c; t d;
#  define __FLT_EORDER6(t, a, b, c, d, e, f)  \
			t a; t b; t c; t d; t e; t f;
#  define __FLT_EORDER7(t, a, b, c, d, e, f, g)  \
			t a; t b; t c; t d; t e; t f; t g;
# else
#  define __FLT_EORDER2(t, a, b) \
			t b; t a;
#  define __FLT_EORDER4(t, a, b, c, d) \
			t d; t c; t b; t a;
#  define __FLT_EORDER6(t, a, b, c, d, e, f)  \
			t f; t e; t d; t c; t b; t a;
#  define __FLT_EORDER7(t, a, b, c, d, e, f, g)  \
			t g; t f; t e; t d; t c; t b; t a;
# endif

/* A union which permits us to convert between _Float128 and
   four 32 bit ints or two 64 bit ints.  */

typedef union
{
  _Float128 value;
  struct
  {
    __FLT_EORDER2 (uint64_t, msw, lsw);
  } parts64;
  struct
  {
    __FLT_EORDER4 (uint32_t, w0, w1, w2, w3);
  } parts32;
} ieee854_float128_shape_type;

/* Get two 64 bit ints from a _Float128.  */

# define GET_FLOAT128_WORDS64(ix0,ix1,d)			\
do {								\
  ieee854_float128_shape_type qw_u;				\
  qw_u.value = (d);						\
  (ix0) = qw_u.parts64.msw;					\
  (ix1) = qw_u.parts64.lsw;					\
} while (0)

/* Set a _Float128 from two 64 bit ints.  */

# define SET_FLOAT128_WORDS64(d,ix0,ix1)			\
do {								\
  ieee854_float128_shape_type qw_u;				\
  qw_u.parts64.msw = (ix0);					\
  qw_u.parts64.lsw = (ix1);					\
  (d) = qw_u.value;						\
} while (0)

/* Get the more significant 64 bits of a _Float128 mantissa.  */

# define GET_FLOAT128_MSW64(v,d)				\
do {								\
  ieee854_float128_shape_type sh_u;				\
  sh_u.value = (d);						\
  (v) = sh_u.parts64.msw;					\
} while (0)

/* Set the more significant 64 bits of a _Float128 mantissa from an int.  */

# define SET_FLOAT128_MSW64(d,v)				\
do {								\
  ieee854_float128_shape_type sh_u;				\
  sh_u.value = (d);						\
  sh_u.parts64.msw = (v);					\
  (d) = sh_u.value;						\
} while (0)

/* Get the least significant 64 bits of a _Float128 mantissa.  */

# define GET_FLOAT128_LSW64(v,d)				\
do {								\
  ieee854_float128_shape_type sh_u;				\
  sh_u.value = (d);						\
  (v) = sh_u.parts64.lsw;					\
} while (0)

/* Likewise, some helper macros which are exposed via ieee754.h for
   C99 real types, but not _Float128.  */

union ieee854_float128
  {
    _Float128 d;

    /* This is the IEEE 854 quad-precision format.  */
    struct
      {
	__FLT_EORDER6 (unsigned int, negative:1,
				     exponent:15,
				     mantissa0:16,
				     mantissa1:32,
				     mantissa2:32,
				     mantissa3:32)
      } ieee;

    /* This format makes it easier to see if a NaN is a signalling NaN.  */
    struct
      {
	__FLT_EORDER7 (unsigned int, negative:1,
				     exponent:15,
				     quiet_nan:1,
				     mantissa0:15,
				     mantissa1:32,
				     mantissa2:32,
				     mantissa3:32)
      } ieee_nan;
  };

#define IEEE854_FLOAT128_BIAS 0x3fff /* Added to exponent.  */

#endif
