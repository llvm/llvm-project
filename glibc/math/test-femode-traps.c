/* Test femode_t functions: test handling of exception traps.
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
test_ee (int exc1, int exc2)
{
  int result = 0;
  printf ("testing %x %x\n", (unsigned int) exc1, (unsigned int) exc2);

  fedisableexcept (FE_ALL_EXCEPT);
  int ret = feenableexcept (exc1);
  if (ret == -1)
    {
      if (EXCEPTION_ENABLE_SUPPORTED (exc1))
	{
	  puts ("first feenableexcept failed unexpectedly");
	  result = 1;
	}
      else
	puts ("first feenableexcept failed, cannot test");
      return result;
    }
  femode_t saved;
  ret = fegetmode (&saved);
  if (ret != 0)
    {
      puts ("fegetmode failed");
      result = 1;
      return result;
    }
  fedisableexcept (FE_ALL_EXCEPT);
  ret = feenableexcept (exc2);
  if (ret == -1)
    {
      if (EXCEPTION_ENABLE_SUPPORTED (exc2))
	{
	  puts ("second feenableexcept failed unexpectedly");
	  result = 1;
	}
      else
	puts ("second feenableexcept failed, cannot test");
      return result;
    }
  ret = fesetmode (&saved);
  if (ret != 0)
    {
      puts ("fesetmode failed");
      result = 1;
      return result;
    }
  /* Verify that the set of enabled traps was restored.  */
  ret = fegetexcept ();
  if (ret != exc1)
    {
      printf ("restored enabled traps %x not %x\n", (unsigned int) ret,
	      (unsigned int) exc1);
      result = 1;
    }
  /* Likewise, with default modes.  */
  ret = fesetmode (FE_DFL_MODE);
  if (ret != 0)
    {
      puts ("fesetmode (FE_DFL_MODE) failed");
      result = 1;
      return result;
    }
  ret = fegetexcept ();
  if (ret != 0)
    {
      printf ("FE_DFL_MODE enabled traps %x not 0\n", (unsigned int) ret);
      result = 1;
    }

  return result;
}

static int
test_e (int exc1)
{
  int result = 0;

  result |= test_ee (exc1, 0);
  result |= test_ee (exc1, FE_ALL_EXCEPT);
#ifdef FE_DIVBYZERO
  result |= test_ee (exc1, FE_DIVBYZERO);
#endif
#ifdef FE_INEXACT
  result |= test_ee (exc1, FE_INEXACT);
#endif
#ifdef FE_INVALID
  result |= test_ee (exc1, FE_INVALID);
#endif
#ifdef FE_OVERFLOW
  result |= test_ee (exc1, FE_OVERFLOW);
#endif
#ifdef FE_UNDERFLOW
  result |= test_ee (exc1, FE_UNDERFLOW);
#endif

  return result;
}

static int
do_test (void)
{
  int result = 0;

  result |= test_e (0);
  result |= test_e (FE_ALL_EXCEPT);
#ifdef FE_DIVBYZERO
  result |= test_e (FE_DIVBYZERO);
#endif
#ifdef FE_INEXACT
  result |= test_e (FE_INEXACT);
#endif
#ifdef FE_INVALID
  result |= test_e (FE_INVALID);
#endif
#ifdef FE_OVERFLOW
  result |= test_e (FE_OVERFLOW);
#endif
#ifdef FE_UNDERFLOW
  result |= test_e (FE_UNDERFLOW);
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
