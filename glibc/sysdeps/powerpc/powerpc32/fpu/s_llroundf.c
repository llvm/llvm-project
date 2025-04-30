/* Round float value to long long int.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <stdint.h>
#include <libm-alias-float.h>

/* Round to the nearest integer, with values exactly on a 0.5 boundary
   rounded away from zero, regardless of the current rounding mode.
   If (long long)x, when x is out of range of a long long, clips at
   LLONG_MAX or LLONG_MIN, then this implementation also clips.  */

long long int
__llroundf (float x)
{
  long long xr;
  if (HAVE_PPC_FCTIDZ)
    xr = (long long) x;
  else
    {
      float ax = fabsf (x);
      /* Avoid incorrect exceptions from libgcc conversions (as of GCC
	 5): <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59412>.  */
      if (ax < 0x1p31f)
	xr = (long long int) (long int) x;
      else if (!(ax < 0x1p55f))
	xr = (long long int) (long int) (x * 0x1p-32f) << 32;
      else
	{
	  uint32_t i0;
	  GET_FLOAT_WORD (i0, x);
	  int exponent = ((i0 >> 23) & 0xff) - 0x7f;
	  unsigned long long int mant = (i0 & 0x7fffff) | 0x800000;
	  mant <<= exponent - 23;
	  xr = (long long int) ((i0 & 0x80000000) != 0 ? -mant : mant);
	}
    }
  /* Avoid spurious "inexact" converting LLONG_MAX to float, and from
     subtraction when the result is out of range, by returning early
     for arguments large enough that no rounding is needed.  */
  if (!(fabsf (x) < 0x1p23f))
    return xr;
  float xrf = (float) xr;

  if (x >= 0.0)
    {
      if (x - xrf >= 0.5)
	xr += (long long) ((unsigned long long) xr + 1) > 0;
    }
  else
    {
      if (xrf - x >= 0.5)
	xr -= (long long) ((unsigned long long) xr - 1) < 0;
    }
  return xr;
}
libm_alias_float (__llround, llround)
