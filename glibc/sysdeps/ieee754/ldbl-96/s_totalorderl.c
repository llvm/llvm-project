/* Total order operation.  ldbl-96 version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <nan-high-order-bit.h>
#include <stdint.h>
#include <shlib-compat.h>
#include <first-versions.h>

int
__totalorderl (const long double *x, const long double *y)
{
  int16_t expx, expy;
  uint32_t hx, hy;
  uint32_t lx, ly;
  GET_LDOUBLE_WORDS (expx, hx, lx, *x);
  GET_LDOUBLE_WORDS (expy, hy, ly, *y);
  if (LDBL_MIN_EXP == -16382)
    {
      /* M68K variant: for the greatest exponent, the high mantissa
	 bit is not significant and both values of it are valid, so
	 set it before comparing.  For the Intel variant, only one
	 value of the high mantissa bit is valid for each exponent, so
	 this is not necessary.  */
      if ((expx & 0x7fff) == 0x7fff)
	hx |= 0x80000000;
      if ((expy & 0x7fff) == 0x7fff)
	hy |= 0x80000000;
    }
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
# error not implemented
#endif
  uint32_t x_sign = expx >> 15;
  uint32_t y_sign = expy >> 15;
  expx ^= x_sign >> 17;
  hx ^= x_sign;
  lx ^= x_sign;
  expy ^= y_sign >> 17;
  hy ^= y_sign;
  ly ^= y_sign;
  return expx < expy || (expx == expy && (hx < hy || (hx == hy && lx <= ly)));
}
#ifdef SHARED
# define CONCATX(x, y) x ## y
# define CONCAT(x, y) CONCATX (x, y)
# define UNIQUE_ALIAS(name) CONCAT (name, __COUNTER__)
# define do_symbol(orig_name, name, aliasname)		\
  strong_alias (orig_name, name)			\
  versioned_symbol (libm, name, aliasname, GLIBC_2_31)
# undef weak_alias
# define weak_alias(name, aliasname)			\
  do_symbol (name, UNIQUE_ALIAS (name), aliasname);
#endif
libm_alias_ldouble (__totalorder, totalorder)
#if SHLIB_COMPAT (libm, GLIBC_2_25, GLIBC_2_31)
int
attribute_compat_text_section
__totalorder_compatl (long double x, long double y)
{
  return __totalorderl (&x, &y);
}
#undef do_symbol
#define do_symbol(orig_name, name, aliasname)			\
  strong_alias (orig_name, name)				\
  compat_symbol (libm, name, aliasname,				\
		 CONCAT (FIRST_VERSION_libm_, aliasname))
libm_alias_ldouble (__totalorder_compat, totalorder)
#endif
