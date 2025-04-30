/* Test for ldbl-128ibm strtold overflow to infinity (bug 14551).
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int
test_strtold_value (const char *s, double exp_hi, double exp_lo, int exp_exc,
		    int exp_errno)
{
  int result = 0;
  union { long double ld; double d[2]; } x;
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
  x.ld = strtold (s, NULL);
  int exc = fetestexcept (FE_ALL_EXCEPT);
  int new_errno = errno;
  printf ("strtold (\"%s\") returned (%a, %a), exceptions 0x%x, errno %d\n",
	  s, x.d[0], x.d[1], exc, new_errno);
  if (x.d[0] == exp_hi)
    printf ("PASS: strtold (\"%s\") high == %a\n", s, exp_hi);
  else
    {
      printf ("FAIL: strtold (\"%s\") high == %a\n", s, exp_hi);
      result = 1;
    }
  if (x.d[1] == exp_lo)
    printf ("PASS: strtold (\"%s\") low == %a\n", s, exp_lo);
  else
    {
      printf ("FAIL: strtold (\"%s\") low == %a\n", s, exp_lo);
      result = 1;
    }
  if (exc == exp_exc)
    printf ("PASS: strtold (\"%s\") exceptions 0x%x\n", s, exp_exc);
  else
    {
      printf ("FAIL: strtold (\"%s\") exceptions 0x%x\n", s, exp_exc);
      result = 1;
    }
  if (new_errno == exp_errno)
    printf ("PASS: strtold (\"%s\") errno %d\n", s, exp_errno);
  else
    {
      printf ("FAIL: strtold (\"%s\") errno %d\n", s, exp_errno);
      result = 1;
    }
  return result;
}

static int
do_test (void)
{
  int result = 0;
  result |= test_strtold_value ("0x1.fffffffffffff8p+1023", INFINITY, 0,
				FE_OVERFLOW | FE_INEXACT, ERANGE);
  result |= test_strtold_value ("-0x1.fffffffffffff8p+1023", -INFINITY, 0,
				FE_OVERFLOW | FE_INEXACT, ERANGE);
  result |= test_strtold_value ("0x1.ffffffffffffffp+1023", INFINITY, 0,
				FE_OVERFLOW | FE_INEXACT, ERANGE);
  result |= test_strtold_value ("-0x1.ffffffffffffffp+1023", -INFINITY, 0,
				FE_OVERFLOW | FE_INEXACT, ERANGE);
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../../../test-skeleton.c"
