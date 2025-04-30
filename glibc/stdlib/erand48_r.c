/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, August 1995.

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

#include <ieee754.h>
#include <stdlib.h>
#include <limits.h>


int
__erand48_r (unsigned short int xsubi[3], struct drand48_data *buffer,
	     double *result)
{
  union ieee754_double temp;

  /* Compute next state.  */
  if (__drand48_iterate (xsubi, buffer) < 0)
    return -1;

  /* Construct a positive double with the 48 random bits distributed over
     its fractional part so the resulting FP number is [0.0,1.0).  */

  temp.ieee.negative = 0;
  temp.ieee.exponent = IEEE754_DOUBLE_BIAS;
  temp.ieee.mantissa0 = (xsubi[2] << 4) | (xsubi[1] >> 12);
  temp.ieee.mantissa1 = ((xsubi[1] & 0xfff) << 20) | (xsubi[0] << 4);

  /* Please note the lower 4 bits of mantissa1 are always 0.  */
  *result = temp.d - 1.0;

  return 0;
}
weak_alias (__erand48_r, erand48_r)
