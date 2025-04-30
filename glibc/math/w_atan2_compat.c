/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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

/*
 * wrapper atan2(y,x)
 */

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-double.h>


#if LIBM_SVID_COMPAT
double
__atan2 (double y, double x)
{
  double z;

  if (__builtin_expect (x == 0.0 && y == 0.0, 0) && _LIB_VERSION == _SVID_)
    return __kernel_standard (y, x, 3); /* atan2(+-0,+-0) */

  z = __ieee754_atan2 (y, x);
  if (__glibc_unlikely (z == 0.0 && y != 0.0 && isfinite (x)))
    __set_errno (ERANGE);
  return z;
}
libm_alias_double (__atan2, atan2)
#endif
