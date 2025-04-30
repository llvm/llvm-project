/* Test totalorderl and totalordermagl for ldbl-128ibm.
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

#include <math.h>
#include <math_ldbl.h>
#include <stdbool.h>
#include <stdio.h>

struct test
{
  double hi, lo1, lo2;
};

static const struct test tests[] =
  {
    { __builtin_nan (""), 1, __builtin_nans ("") },
    { -__builtin_nan (""), 1, __builtin_nans ("") },
    { __builtin_nans (""), 1, __builtin_nan ("") },
    { -__builtin_nans (""), 1, __builtin_nan ("") },
    { __builtin_inf (), 0.0, -0.0 },
    { -__builtin_inf (), 0.0, -0.0 },
    { 1.5, 0.0, -0.0 },
  };

static int
do_test (void)
{
  int result = 0;

  for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); i++)
    {
      long double ldx = ldbl_pack (tests[i].hi, tests[i].lo1);
      long double ldy = ldbl_pack (tests[i].hi, tests[i].lo2);
      bool to1 = totalorderl (&ldx, &ldy);
      bool to2 = totalorderl (&ldy, &ldx);
      if (to1 && to2)
	printf ("PASS: test %zu\n", i);
      else
	{
	  printf ("FAIL: test %zu\n", i);
	  result = 1;
	}
      to1 = totalordermagl (&ldx, &ldy);
      to2 = totalordermagl (&ldy, &ldx);
      if (to1 && to2)
	printf ("PASS: test %zu (totalordermagl)\n", i);
      else
	{
	  printf ("FAIL: test %zu (totalordermagl)\n", i);
	  result = 1;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
