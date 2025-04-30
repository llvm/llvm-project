/* Test totalorderl and totalordermagl for ldbl-96.
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

static const uint64_t tests[] =
  {
    0, 1, 0x4000000000000000ULL, 0x4000000000000001ULL,
    0x7fffffffffffffffULL
  };

static int
do_test (void)
{
  int result = 0;

  if (LDBL_MIN_EXP == -16382)
    for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); i++)
      {
	long double ldx, ldy, ldnx, ldny;
	/* Verify that the high bit of the mantissa is ignored for
	   infinities and NaNs for the M68K variant of this
	   format.  */
	SET_LDOUBLE_WORDS (ldx, 0x7fff,
			   tests[i] >> 32, tests[i] & 0xffffffffULL);
	SET_LDOUBLE_WORDS (ldy, 0x7fff,
			   (tests[i] >> 32) | 0x80000000,
			   tests[i] & 0xffffffffULL);
	SET_LDOUBLE_WORDS (ldnx, 0xffff,
			   tests[i] >> 32, tests[i] & 0xffffffffULL);
	SET_LDOUBLE_WORDS (ldny, 0xffff,
			   (tests[i] >> 32) | 0x80000000,
			   tests[i] & 0xffffffffULL);
	bool to1 = totalorderl (&ldx, &ldy);
	bool to2 = totalorderl (&ldy, &ldx);
	bool to3 = totalorderl (&ldnx, &ldny);
	bool to4 = totalorderl (&ldny, &ldnx);
	if (to1 && to2 && to3 && to4)
	  printf ("PASS: test %zu\n", i);
	else
	  {
	    printf ("FAIL: test %zu\n", i);
	    result = 1;
	  }
	to1 = totalordermagl (&ldx, &ldy);
	to2 = totalordermagl (&ldy, &ldx);
	to3 = totalordermagl (&ldnx, &ldny);
	to4 = totalordermagl (&ldny, &ldnx);
	if (to1 && to2 && to3 && to4)
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
