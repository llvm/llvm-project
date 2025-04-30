/* Total order operation.  flt-32 version.
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

#include <math.h>
#include <math_private.h>
#include <libm-alias-float.h>
#include <nan-high-order-bit.h>
#include <stdint.h>
#include <shlib-compat.h>
#include <first-versions.h>

int
__totalorderf (const float *x, const float *y)
{
  int32_t ix, iy;
  GET_FLOAT_WORD (ix, *x);
  GET_FLOAT_WORD (iy, *y);
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
  /* For the preferred quiet NaN convention, this operation is a
     comparison of the representations of the arguments interpreted as
     sign-magnitude integers.  If both arguments are NaNs, invert the
     quiet/signaling bit so comparing that way works.  */
  if ((ix & 0x7fffffff) > 0x7f800000 && (iy & 0x7fffffff) > 0x7f800000)
    {
      ix ^= 0x00400000;
      iy ^= 0x00400000;
    }
#endif
  uint32_t ix_sign = ix >> 31;
  uint32_t iy_sign = iy >> 31;
  ix ^= ix_sign >> 1;
  iy ^= iy_sign >> 1;
  return ix <= iy;
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
libm_alias_float (__totalorder, totalorder)
#if SHLIB_COMPAT (libm, GLIBC_2_25, GLIBC_2_31)
int
attribute_compat_text_section
__totalorder_compatf (float x, float y)
{
  return __totalorderf (&x, &y);
}
#undef do_symbol
#define do_symbol(orig_name, name, aliasname)			\
  strong_alias (orig_name, name)				\
  compat_symbol (libm, name, aliasname,				\
		 CONCAT (FIRST_VERSION_libm_, aliasname))
libm_alias_float (__totalorder_compat, totalorder)
#endif
