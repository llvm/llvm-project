/* Test that tininess.h is correct for this architecture.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <float.h>
#include <stdio.h>
#include <tininess.h>

volatile float a = 0x1.fffp-126;
volatile float b = 0x1.0008p-1;
volatile float c;
volatile float m = FLT_MIN;
volatile float mm;

static int
do_test (void)
{
  int result = 0;
#ifdef FE_UNDERFLOW
  feclearexcept (FE_ALL_EXCEPT);
  mm = m * m;
  if (!fetestexcept (FE_UNDERFLOW))
    {
      puts ("underflow exception not supported at runtime, cannot test");
      return 0;
    }
  feclearexcept (FE_ALL_EXCEPT);
  c = a * b;
  if (fetestexcept (FE_UNDERFLOW))
    {
      if (TININESS_AFTER_ROUNDING)
	{
	  puts ("tininess.h says after rounding, "
		"but detected before rounding");
	  result = 1;
	}
    }
  else
    {
      if (!TININESS_AFTER_ROUNDING)
	{
	  puts ("tininess.h says before rounding, "
		"but detected after rounding");
	  result = 1;
	}
    }
#else
  puts ("underflow exception not supported at compile time, cannot test");
#endif
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
