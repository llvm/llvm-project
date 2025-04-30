/* Multiply by integral power of radix.

   Copyright (C) 2011-2021 Free Software Foundation, Inc.

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
#include <libm-alias-finite.h>

static FLOAT
__attribute__ ((noinline))
invalid_fn (FLOAT x, FLOAT fn)
{
  if (M_SUF (rint) (fn) != fn)
    return (fn - fn) / (fn - fn);
  else if (fn > M_LIT (65000.0))
    return M_SUF (__scalbn) (x, 65000);
  else
    return M_SUF (__scalbn) (x,-65000);
}


FLOAT
M_DECL_FUNC (__ieee754_scalb) (FLOAT x, FLOAT fn)
{
  if (__glibc_unlikely (isnan (x)))
    return x * fn;
  if (__glibc_unlikely (!isfinite (fn)))
    {
      if (isnan (fn) || fn > M_LIT (0.0))
	return x * fn;
      if (x == M_LIT (0.0))
	return x;
      return x / -fn;
    }
  if (__glibc_unlikely (M_FABS (fn) >= M_LIT (0x1p31)
			|| (FLOAT) (int) fn != fn))
    return invalid_fn (x, fn);

  return M_SCALBN (x, (int) fn);
}
declare_mgen_finite_alias (__ieee754_scalb, __scalb)
