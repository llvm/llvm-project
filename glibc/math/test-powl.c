/* Test for powl
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

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <ieee754.h>

static int
do_test (void)
{
  int result = 0;

#if LDBL_MANT_DIG == 64
    {
      long double x = 1e-20;
      union ieee854_long_double u;
      u.ieee.mantissa0 = 1;
      u.ieee.mantissa1 = 1;
      u.ieee.exponent = 0;
      u.ieee.negative = 0;
      (void) powl (0.2, u.d);
      x = powl (x, 1.5);
      if (fabsl (x - 1e-30) > 1e-10)
	{
	  printf ("powl (1e-20, 1.5): wrong result: %Lg\n", x);
	  result = 1;
	}
    }
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
