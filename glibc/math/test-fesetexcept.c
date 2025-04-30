/* Test fesetexcept.
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

#include <fenv.h>
#include <stdio.h>
#include <math-tests.h>

static int
test_fesetexcept (int exc, const char *exc_name)
{
  int result = 0;

  printf ("Testing %s\n", exc_name);
  feclearexcept (FE_ALL_EXCEPT);
  int ret = fesetexcept (exc);
  if (ret == 0)
    printf ("fesetexcept (%s) succeeded\n", exc_name);
  else
    {
      printf ("fesetexcept (%s) failed\n", exc_name);
      if (exc == 0 || EXCEPTION_TESTS (float))
	{
	  puts ("failure of fesetexcept was unexpected");
	  result = 1;
	}
      else
	puts ("failure of fesetexcept OK, skipping further tests");
      return result;
    }
  ret = fetestexcept (FE_ALL_EXCEPT);
  if (ret != exc)
    {
      printf ("raised exceptions %x, expected %x\n",
	      (unsigned int) ret, (unsigned int) exc);
      result = 1;
    }

  ret = feraiseexcept (FE_ALL_EXCEPT);
  if (ret != 0)
    {
      if (exc == 0 && !EXCEPTION_TESTS (float))
	{
	  puts ("feraiseexcept (FE_ALL_EXCEPT) failed, skipping further tests");
	  return result;
	}
      puts ("feraiseexcept (FE_ALL_EXCEPT) unexpectedly failed");
      result = 1;
    }
  ret = fesetexcept (exc);
  if (ret != 0)
    {
      puts ("fesetexcept (second test) unexpectedly failed");
      result = 1;
    }
  ret = fetestexcept (FE_ALL_EXCEPT);
  if (ret != FE_ALL_EXCEPT)
    {
      printf ("raised exceptions (second test) %x, expected %x\n",
	      (unsigned int) ret, (unsigned int) FE_ALL_EXCEPT);
      result = 1;
    }

  feclearexcept (FE_ALL_EXCEPT);
  ret = feraiseexcept (FE_ALL_EXCEPT & ~exc);
  if (ret != 0)
    {
      puts ("feraiseexcept (third test) unexpectedly failed");
      result = 1;
    }
  ret = fesetexcept (exc);
  if (ret != 0)
    {
      puts ("fesetexcept (third test) unexpectedly failed");
      result = 1;
    }
  ret = fetestexcept (FE_ALL_EXCEPT);
  if (ret != FE_ALL_EXCEPT)
    {
      printf ("raised exceptions (third test) %x, expected %x\n",
	      (unsigned int) ret, (unsigned int) FE_ALL_EXCEPT);
      result = 1;
    }

  return result;
}

static int
do_test (void)
{
  int result = 0;

  result |= test_fesetexcept (0, "0");
  result |= test_fesetexcept (FE_ALL_EXCEPT, "FE_ALL_EXCEPT");
#ifdef FE_DIVBYZERO
  result |= test_fesetexcept (FE_DIVBYZERO, "FE_DIVBYZERO");
#endif
#ifdef FE_INEXACT
  result |= test_fesetexcept (FE_INEXACT, "FE_INEXACT");
#endif
#ifdef FE_INVALID
  result |= test_fesetexcept (FE_INVALID, "FE_INVALID");
#endif
#ifdef FE_OVERFLOW
  result |= test_fesetexcept (FE_OVERFLOW, "FE_OVERFLOW");
#endif
#ifdef FE_UNDERFLOW
  result |= test_fesetexcept (FE_UNDERFLOW, "FE_UNDERFLOW");
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
