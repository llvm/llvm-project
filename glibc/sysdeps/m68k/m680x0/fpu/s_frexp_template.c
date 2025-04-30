/* Implement frexp for m68k.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

FLOAT
M_DECL_FUNC (__frexp) (FLOAT value, int *expptr)
{
  FLOAT mantissa, exponent;
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
  __asm ("fscale%.l %2, %0"
	 : "=f" (mantissa)
	 : "0" (value), "dmi" (-iexponent));
  return mantissa;
}
declare_mgen_alias (__frexp, frexp)
