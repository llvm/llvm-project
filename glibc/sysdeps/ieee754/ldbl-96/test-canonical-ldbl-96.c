/* Test iscanonical and canonicalizel for ldbl-96.
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
#include <stdint.h>
#include <stdio.h>

struct test
{
  bool sign;
  uint16_t exponent;
  bool high;
  uint64_t mantissa;
  bool canonical;
};

#define M68K_VARIANT (LDBL_MIN_EXP == -16382)

static const struct test tests[] =
  {
    { false, 0, true, 0, M68K_VARIANT },
    { true, 0, true, 0, M68K_VARIANT },
    { false, 0, true, 1, M68K_VARIANT },
    { true, 0, true, 1, M68K_VARIANT },
    { false, 0, true, 0x100000000ULL, M68K_VARIANT },
    { true, 0, true, 0x100000000ULL, M68K_VARIANT },
    { false, 0, false, 0, true },
    { true, 0, false, 0, true },
    { false, 0, false, 1, true },
    { true, 0, false, 1, true },
    { false, 0, false, 0x100000000ULL, true },
    { true, 0, false, 0x100000000ULL, true },
    { false, 1, true, 0, true },
    { true, 1, true, 0, true },
    { false, 1, true, 1, true },
    { true, 1, true, 1, true },
    { false, 1, true, 0x100000000ULL, true },
    { true, 1, true, 0x100000000ULL, true },
    { false, 1, false, 0, false },
    { true, 1, false, 0, false },
    { false, 1, false, 1, false },
    { true, 1, false, 1, false },
    { false, 1, false, 0x100000000ULL, false },
    { true, 1, false, 0x100000000ULL, false },
    { false, 0x7ffe, true, 0, true },
    { true, 0x7ffe, true, 0, true },
    { false, 0x7ffe, true, 1, true },
    { true, 0x7ffe, true, 1, true },
    { false, 0x7ffe, true, 0x100000000ULL, true },
    { true, 0x7ffe, true, 0x100000000ULL, true },
    { false, 0x7ffe, false, 0, false },
    { true, 0x7ffe, false, 0, false },
    { false, 0x7ffe, false, 1, false },
    { true, 0x7ffe, false, 1, false },
    { false, 0x7ffe, false, 0x100000000ULL, false },
    { true, 0x7ffe, false, 0x100000000ULL, false },
    { false, 0x7fff, true, 0, true },
    { true, 0x7fff, true, 0, true },
    { false, 0x7fff, true, 1, true },
    { true, 0x7fff, true, 1, true },
    { false, 0x7fff, true, 0x100000000ULL, true },
    { true, 0x7fff, true, 0x100000000ULL, true },
    { false, 0x7fff, false, 0, M68K_VARIANT },
    { true, 0x7fff, false, 0, M68K_VARIANT },
    { false, 0x7fff, false, 1, M68K_VARIANT },
    { true, 0x7fff, false, 1, M68K_VARIANT },
    { false, 0x7fff, false, 0x100000000ULL, M68K_VARIANT },
    { true, 0x7fff, false, 0x100000000ULL, M68K_VARIANT },
  };

static int
do_test (void)
{
  int result = 0;

  for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); i++)
    {
      long double ld;
      SET_LDOUBLE_WORDS (ld, tests[i].exponent | (tests[i].sign << 15),
			 (tests[i].mantissa >> 32) | (tests[i].high << 31),
			 tests[i].mantissa & 0xffffffffULL);
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
