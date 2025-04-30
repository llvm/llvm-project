/* Test iseqsig with excess precision.
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
#include <stdio.h>

static int
do_test (void)
{
  int result = 0;

  if (FLT_EVAL_METHOD == 1 || FLT_EVAL_METHOD == 2 || FLT_EVAL_METHOD > 32)
    {
      /* Excess precision for float.  */
      if (iseqsig (1.0f, 1.0f + (float) DBL_EPSILON))
	{
	  puts ("iseqsig removes excess precision float -> double");
	  result = 1;
	}
      else
	puts ("iseqsig preserves excess precision float -> double");
      if (iseqsig (__builtin_inff (), FLT_MAX * FLT_MAX))
	{
	  puts ("iseqsig removes excess range float -> double");
	  result = 1;
	}
      else
	puts ("iseqsig preserves excess range float -> double");
    }

  if (FLT_EVAL_METHOD == 2 || FLT_EVAL_METHOD > 64)
    {
      /* Excess precision for float and double.  */
      if (iseqsig (1.0f, 1.0f + (float) LDBL_EPSILON))
	{
	  puts ("iseqsig removes excess precision float -> long double");
	  result = 1;
	}
      else
	puts ("iseqsig preserves excess precision float -> long double");
      if (iseqsig (1.0, 1.0 + (double) LDBL_EPSILON))
	{
	  puts ("iseqsig removes excess precision double -> long double");
	  result = 1;
	}
      else
	puts ("iseqsig preserves excess precision double -> long double");
      if (LDBL_MAX_EXP >= 2 * DBL_MAX_EXP)
	{
	  if (iseqsig (__builtin_inf (), DBL_MAX * DBL_MAX))
	    {
	      puts ("iseqsig removes excess range double -> long double");
	      result = 1;
	    }
	    else
	      puts ("iseqsig preserves excess range double -> long double");
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
