/* Test fetestexceptflag.
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
test_one (int exc_test, int exc_set, int exc_save)
{
  int result = 0;

  printf ("Individual test: %x %x %x\n", (unsigned int) exc_test,
	  (unsigned int) exc_set, (unsigned int) exc_save);

  feclearexcept (FE_ALL_EXCEPT);
  int ret = fesetexcept (exc_set);
  if (ret != 0)
    {
      puts ("fesetexcept failed");
      if (exc_set == 0 || EXCEPTION_TESTS (float))
	{
	  puts ("failure of fesetexcept was unexpected");
	  result = 1;
	}
      else
	puts ("failure of fesetexcept OK, skipping further tests");
      return result;
    }
  fexcept_t saved;
  ret = fegetexceptflag (&saved, exc_save);
  if (ret == 0)
    puts ("fegetexceptflag succeeded");
  else
    {
      puts ("fegetexceptflag failed");
      result = 1;
      return result;
    }
  ret = fetestexceptflag (&saved, exc_test);
  if (ret == (exc_set & exc_test))
    puts ("fetestexceptflag result correct");
  else
    {
      printf ("fetestexceptflag returned %x, expected %x\n", ret,
	      exc_set & exc_test);
      result = 1;
    }
  if (exc_save == FE_ALL_EXCEPT)
    {
      /* Also test fetestexceptflag testing all exceptions but
	 possibly with only some set.  */
      ret = fetestexceptflag (&saved, FE_ALL_EXCEPT);
      if (ret == exc_set)
	puts ("fetestexceptflag (FE_ALL_EXCEPT) result correct");
      else
	{
	  printf ("fetestexceptflag (FE_ALL_EXCEPT) returned %x, expected %x\n",
		  ret, exc_set);
	  result = 1;
	}
    }
  return result;
}

static int
test_fetestexceptflag (int exc, const char *exc_name)
{
  int result = 0;

  printf ("Testing %s\n", exc_name);

  /* Test each case of: whether this exception is set or clear;
     whether other exceptions are set or clear; whether the whole
     state is saved or just the state for this exception.  */
  result |= test_one (exc, 0, exc);
  result |= test_one (exc, 0, FE_ALL_EXCEPT);
  result |= test_one (exc, exc, exc);
  result |= test_one (exc, exc, FE_ALL_EXCEPT);
  result |= test_one (exc, FE_ALL_EXCEPT & ~exc, exc);
  result |= test_one (exc, FE_ALL_EXCEPT & ~exc, FE_ALL_EXCEPT);
  result |= test_one (exc, FE_ALL_EXCEPT, exc);
  result |= test_one (exc, FE_ALL_EXCEPT, FE_ALL_EXCEPT);

  return result;
}

static int
do_test (void)
{
  int result = 0;

  result |= test_fetestexceptflag (0, "0");
  result |= test_fetestexceptflag (FE_ALL_EXCEPT, "FE_ALL_EXCEPT");
#ifdef FE_DIVBYZERO
  result |= test_fetestexceptflag (FE_DIVBYZERO, "FE_DIVBYZERO");
#endif
#ifdef FE_INEXACT
  result |= test_fetestexceptflag (FE_INEXACT, "FE_INEXACT");
#endif
#ifdef FE_INVALID
  result |= test_fetestexceptflag (FE_INVALID, "FE_INVALID");
#endif
#ifdef FE_OVERFLOW
  result |= test_fetestexceptflag (FE_OVERFLOW, "FE_OVERFLOW");
#endif
#ifdef FE_UNDERFLOW
  result |= test_fetestexceptflag (FE_UNDERFLOW, "FE_UNDERFLOW");
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
