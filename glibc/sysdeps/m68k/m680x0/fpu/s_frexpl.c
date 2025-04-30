/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>

long double
__frexpl (long double value, int *expptr)
{
  long double mantissa, exponent;
  int iexponent;
  unsigned long fpsr;

  __asm ("ftst%.x %1\n"
	 "fmove%.l %/fpsr, %0"
	 : "=dm" (fpsr) : "f" (value));
  if (fpsr & (7 << 24))
    {
      /* Not finite or zero.  */
      *expptr = 0;
      return value;
    }
  __asm ("fgetexp%.x %1, %0" : "=f" (exponent) : "f" (value));
  iexponent = (int) exponent + 1;
  *expptr = iexponent;
  /* Unnormalized numbers must be handled specially, otherwise fscale
     results in overflow.  */
  if (iexponent <= -16384)
    {
      value *= 0x1p16383L;
      iexponent += 16383;
    }
  else if (iexponent >= 16384)
    {
      value *= 0x1p-16383L;
      iexponent -= 16383;
    }

  __asm ("fscale%.l %2, %0"
	 : "=f" (mantissa)
	 : "0" (value), "dmi" (-iexponent));
  return mantissa;
}

weak_alias (__frexpl, frexpl)
