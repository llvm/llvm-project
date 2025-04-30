/* Test iscanonical and canonicalizel for ldbl-128ibm.
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
#include <math_ldbl.h>
#include <stdbool.h>
#include <stdio.h>

struct test
{
  double hi, lo;
  bool canonical;
};

static const struct test tests[] =
  {
    { __builtin_nan (""), 0.0, true },
    { __builtin_nan (""), DBL_MAX, true },
    { __builtin_nan (""), __builtin_inf (), true },
    { __builtin_nan (""), __builtin_nan (""), true },
    { __builtin_nan (""), __builtin_nans (""), true },
    { __builtin_nans (""), 0.0, true },
    { __builtin_nans (""), DBL_MAX, true },
    { __builtin_nans (""), __builtin_inf (), true },
    { __builtin_nans (""), __builtin_nan (""), true },
    { __builtin_nans (""), __builtin_nans (""), true },
    { __builtin_inf (), 0.0, true },
    { __builtin_inf (), -0.0, true },
    { -__builtin_inf (), 0.0, true },
    { -__builtin_inf (), -0.0, true },
    { __builtin_inf (), DBL_TRUE_MIN, false },
    { __builtin_inf (), -DBL_TRUE_MIN, false },
    { -__builtin_inf (), DBL_TRUE_MIN, false },
    { -__builtin_inf (), -DBL_TRUE_MIN, false },
    { __builtin_inf (), DBL_MIN, false },
    { __builtin_inf (), -DBL_MIN, false },
    { -__builtin_inf (), DBL_MIN, false },
    { -__builtin_inf (), -DBL_MIN, false },
    { __builtin_inf (), __builtin_inf (), false },
    { __builtin_inf (), -__builtin_inf (), false },
    { -__builtin_inf (), __builtin_inf (), false },
    { -__builtin_inf (), -__builtin_inf (), false },
    { __builtin_inf (), __builtin_nan (""), false },
    { __builtin_inf (), -__builtin_nan (""), false },
    { -__builtin_inf (), __builtin_nan (""), false },
    { -__builtin_inf (), -__builtin_nan (""), false },
    { 0.0, 0.0, true },
    { 0.0, -0.0, true },
    { -0.0, 0.0, true },
    { -0.0, -0.0, true },
    { 0.0, DBL_TRUE_MIN, false },
    { 0.0, -DBL_TRUE_MIN, false },
    { -0.0, DBL_TRUE_MIN, false },
    { -0.0, -DBL_TRUE_MIN, false },
    { 0.0, DBL_MAX, false },
    { 0.0, -DBL_MAX, false },
    { -0.0, DBL_MAX, false },
    { -0.0, -DBL_MAX, false },
    { 0.0, __builtin_inf (), false },
    { 0.0, -__builtin_inf (), false },
    { -0.0, __builtin_inf (), false },
    { -0.0, -__builtin_inf (), false },
    { 0.0, __builtin_nan (""), false },
    { 0.0, -__builtin_nan (""), false },
    { -0.0, __builtin_nan (""), false },
    { -0.0, -__builtin_nan (""), false },
    { 1.0, 0.0, true },
    { 1.0, -0.0, true },
    { -1.0, 0.0, true },
    { -1.0, -0.0, true },
    { 1.0, DBL_TRUE_MIN, true },
    { 1.0, -DBL_TRUE_MIN, true },
    { -1.0, DBL_TRUE_MIN, true },
    { -1.0, -DBL_TRUE_MIN, true },
    { 1.0, DBL_MAX, false },
    { 1.0, -DBL_MAX, false },
    { -1.0, DBL_MAX, false },
    { -1.0, -DBL_MAX, false },
    { 1.0, __builtin_inf (), false },
    { 1.0, -__builtin_inf (), false },
    { -1.0, __builtin_inf (), false },
    { -1.0, -__builtin_inf (), false },
    { 1.0, __builtin_nan (""), false },
    { 1.0, -__builtin_nan (""), false },
    { -1.0, __builtin_nan (""), false },
    { -1.0, -__builtin_nan (""), false },
    { 0x1p1023, 0x1.1p969, true },
    { 0x1p1023, -0x1.1p969, true },
    { -0x1p1023, 0x1.1p969, true },
    { -0x1p1023, -0x1.1p969, true },
    { 0x1p1023, 0x1.1p970, false },
    { 0x1p1023, -0x1.1p970, false },
    { -0x1p1023, 0x1.1p970, false },
    { -0x1p1023, -0x1.1p970, false },
    { 0x1p1023, 0x1p970, true },
    { 0x1p1023, -0x1p970, true },
    { -0x1p1023, 0x1p970, true },
    { -0x1p1023, -0x1p970, true },
    { 0x1.0000000000001p1023, 0x1p970, false },
    { 0x1.0000000000001p1023, -0x1p970, false },
    { -0x1.0000000000001p1023, 0x1p970, false },
    { -0x1.0000000000001p1023, -0x1p970, false },
    { 0x1p-969, 0x1.1p-1023, true },
    { 0x1p-969, -0x1.1p-1023, true },
    { -0x1p-969, 0x1.1p-1023, true },
    { -0x1p-969, -0x1.1p-1023, true },
    { 0x1p-969, 0x1.1p-1022, false },
    { 0x1p-969, -0x1.1p-1022, false },
    { -0x1p-969, 0x1.1p-1022, false },
    { -0x1p-969, -0x1.1p-1022, false },
    { 0x1p-969, 0x1p-1022, true },
    { 0x1p-969, -0x1p-1022, true },
    { -0x1p-969, 0x1p-1022, true },
    { -0x1p-969, -0x1p-1022, true },
    { 0x1.0000000000001p-969, 0x1p-1022, false },
    { 0x1.0000000000001p-969, -0x1p-1022, false },
    { -0x1.0000000000001p-969, 0x1p-1022, false },
    { -0x1.0000000000001p-969, -0x1p-1022, false },
    { 0x1p-970, 0x1.1p-1024, true },
    { 0x1p-970, -0x1.1p-1024, true },
    { -0x1p-970, 0x1.1p-1024, true },
    { -0x1p-970, -0x1.1p-1024, true },
    { 0x1p-970, 0x1.1p-1023, false },
    { 0x1p-970, -0x1.1p-1023, false },
    { -0x1p-970, 0x1.1p-1023, false },
    { -0x1p-970, -0x1.1p-1023, false },
    { 0x1p-970, 0x1p-1023, true },
    { 0x1p-970, -0x1p-1023, true },
    { -0x1p-970, 0x1p-1023, true },
    { -0x1p-970, -0x1p-1023, true },
    { 0x1.0000000000001p-970, 0x1p-1023, false },
    { 0x1.0000000000001p-970, -0x1p-1023, false },
    { -0x1.0000000000001p-970, 0x1p-1023, false },
    { -0x1.0000000000001p-970, -0x1p-1023, false },
    { 0x1p-1000, 0x1.1p-1054, true },
    { 0x1p-1000, -0x1.1p-1054, true },
    { -0x1p-1000, 0x1.1p-1054, true },
    { -0x1p-1000, -0x1.1p-1054, true },
    { 0x1p-1000, 0x1.1p-1053, false },
    { 0x1p-1000, -0x1.1p-1053, false },
    { -0x1p-1000, 0x1.1p-1053, false },
    { -0x1p-1000, -0x1.1p-1053, false },
    { 0x1p-1000, 0x1p-1053, true },
    { 0x1p-1000, -0x1p-1053, true },
    { -0x1p-1000, 0x1p-1053, true },
    { -0x1p-1000, -0x1p-1053, true },
    { 0x1.0000000000001p-1000, 0x1p-1053, false },
    { 0x1.0000000000001p-1000, -0x1p-1053, false },
    { -0x1.0000000000001p-1000, 0x1p-1053, false },
    { -0x1.0000000000001p-1000, -0x1p-1053, false },
    { 0x1p-1021, 0x1p-1074, true },
    { 0x1p-1021, -0x1p-1074, true },
    { -0x1p-1021, 0x1p-1074, true },
    { -0x1p-1021, -0x1p-1074, true },
    { 0x1.0000000000001p-1021, 0x1p-1074, false },
    { 0x1.0000000000001p-1021, -0x1p-1074, false },
    { -0x1.0000000000001p-1021, 0x1p-1074, false },
    { -0x1.0000000000001p-1021, -0x1p-1074, false },
    { 0x1p-1022, 0x1p-1074, false },
    { 0x1p-1022, -0x1p-1074, false },
    { -0x1p-1022, 0x1p-1074, false },
    { -0x1p-1022, -0x1p-1074, false },
  };

static int
do_test (void)
{
  int result = 0;

  for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); i++)
    {
      long double ld = ldbl_pack (tests[i].hi, tests[i].lo);
      bool canonical = iscanonical (ld);
      if (canonical == tests[i].canonical)
	{
	  printf ("PASS: iscanonical test %zu\n", i);
	  long double ldc = 12345.0L;
	  bool canonicalize_ret = canonicalizel (&ldc, &ld);
	  if (canonicalize_ret == !canonical)
	    {
	      printf ("PASS: canonicalizel test %zu\n", i);
	      bool canon_ok;
	      if (!canonical)
		canon_ok = ldc == 12345.0L;
	      else if (isnan (ld))
		canon_ok = isnan (ldc) && !issignaling (ldc);
	      else
		canon_ok = ldc == ld;
	      if (canon_ok)
		printf ("PASS: canonicalized value test %zu\n", i);
	      else
		{
		  printf ("FAIL: canonicalized value test %zu\n", i);
		  result = 1;
		}
	    }
	  else
	    {
	      printf ("FAIL: canonicalizel test %zu\n", i);
	      result = 1;
	    }
	}
      else
	{
	  printf ("FAIL: iscanonical test %zu\n", i);
	  result = 1;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
