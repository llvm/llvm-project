/* Test fegetexceptflag and fesetexceptflag.
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

/* Like feraiseexcept, but raise exactly the specified exceptions EXC,
   without possibly raising "inexact" together with "overflow" or
   "underflow" as permitted by ISO C.  (This is not used with traps
   enabled, so side-effects from raising and then clearing "inexact"
   are irrelevant.)  */

static int
feraiseexcept_exact (int exc)
{
#ifdef FE_INEXACT
  int mask = 0;
#ifdef FE_OVERFLOW
  mask |= FE_OVERFLOW;
#endif
#ifdef FE_UNDERFLOW
  mask |= FE_UNDERFLOW;
#endif
  if ((exc & FE_INEXACT) != 0
      || (exc & mask) == 0
      || fetestexcept (FE_INEXACT) != 0)
    return feraiseexcept (exc);
  int ret = feraiseexcept (exc);
  feclearexcept (FE_INEXACT);
  return ret;
#else
  return feraiseexcept (exc);
#endif
}

static int
test_set (int initial, const fexcept_t *saved, int mask, int expected)
{
  int result = 0;
  feclearexcept (FE_ALL_EXCEPT);
  printf ("Testing set: initial exceptions %x, mask %x, expected %x\n",
	  (unsigned int) initial, (unsigned int) mask,
	  (unsigned int) expected);
  int ret = feraiseexcept_exact (initial);
  if (ret != 0)
    {
      puts ("feraiseexcept failed");
      if (initial == 0 || EXCEPTION_TESTS (float))
	{
	  puts ("failure of feraiseexcept was unexpected");
	  result = 1;
	}
      else
	puts ("failure of feraiseexcept OK, skipping further tests");
      return result;
    }
  ret = fesetexceptflag (saved, mask);
  if (ret != 0)
    {
      puts ("fesetexceptflag failed");
      result = 1;
    }
  else
    puts ("fesetexceptflag succeeded");
  ret = fetestexcept (FE_ALL_EXCEPT);
  if (ret != expected)
    {
      printf ("raised exceptions %x, expected %x\n",
	      (unsigned int) ret, (unsigned int) expected);
      result = 1;
    }
  return result;
}

static int
test_except (int exc, const char *exc_name)
{
  int result = 0;

  printf ("Testing %s\n", exc_name);
  feclearexcept (FE_ALL_EXCEPT);

  fexcept_t clear_saved_exc, clear_saved_all;
  int ret = fegetexceptflag (&clear_saved_exc, exc);
  if (ret == 0)
    printf ("fegetexceptflag (%s) succeeded\n", exc_name);
  else
    {
      printf ("fegetexceptflag (%s) failed\n", exc_name);
      result = 1;
      return result;
    }
  ret = fegetexceptflag (&clear_saved_all, FE_ALL_EXCEPT);
  if (ret == 0)
    puts ("fegetexceptflag (FE_ALL_EXCEPT) succeeded");
  else
    {
      puts ("fegetexceptflag (FE_ALL_EXCEPT) failed");
      result = 1;
      return result;
    }

  ret = feraiseexcept_exact (exc);
  if (ret == 0)
    printf ("feraiseexcept (%s) succeeded\n", exc_name);
  else
    {
      printf ("feraiseexcept (%s) failed\n", exc_name);
      if (exc == 0 || EXCEPTION_TESTS (float))
	{
	  puts ("failure of feraiseexcept was unexpected");
	  result = 1;
	}
      else
	puts ("failure of feraiseexcept OK, skipping further tests");
      return result;
    }

  fexcept_t set_saved_exc, set_saved_all;
  ret = fegetexceptflag (&set_saved_exc, exc);
  if (ret == 0)
    printf ("fegetexceptflag (%s) succeeded\n", exc_name);
  else
    {
      printf ("fegetexceptflag (%s) failed\n", exc_name);
      result = 1;
      return result;
    }
  ret = fegetexceptflag (&set_saved_all, FE_ALL_EXCEPT);
  if (ret == 0)
    puts ("fegetexceptflag (FE_ALL_EXCEPT) succeeded");
  else
    {
      puts ("fegetexceptflag (FE_ALL_EXCEPT) failed");
      result = 1;
      return result;
    }

  result |= test_set (0, &set_saved_exc, exc, exc);
  result |= test_set (0, &set_saved_all, exc, exc);
  result |= test_set (0, &set_saved_all, FE_ALL_EXCEPT, exc);
  result |= test_set (0, &clear_saved_exc, exc, 0);
  result |= test_set (0, &clear_saved_all, exc, 0);
  result |= test_set (0, &clear_saved_all, FE_ALL_EXCEPT, 0);
  result |= test_set (exc, &set_saved_exc, exc, exc);
  result |= test_set (exc, &set_saved_all, exc, exc);
  result |= test_set (exc, &set_saved_all, FE_ALL_EXCEPT, exc);
  result |= test_set (exc, &clear_saved_exc, exc, 0);
  result |= test_set (exc, &clear_saved_all, exc, 0);
  result |= test_set (exc, &clear_saved_all, FE_ALL_EXCEPT, 0);
  result |= test_set (FE_ALL_EXCEPT, &set_saved_exc, exc, FE_ALL_EXCEPT);
  result |= test_set (FE_ALL_EXCEPT, &set_saved_all, exc, FE_ALL_EXCEPT);
  result |= test_set (FE_ALL_EXCEPT, &set_saved_all, FE_ALL_EXCEPT, exc);
  result |= test_set (FE_ALL_EXCEPT, &clear_saved_exc, exc,
		      FE_ALL_EXCEPT & ~exc);
  result |= test_set (FE_ALL_EXCEPT, &clear_saved_all, exc,
		      FE_ALL_EXCEPT & ~exc);
  result |= test_set (FE_ALL_EXCEPT, &clear_saved_all, FE_ALL_EXCEPT, 0);

  return result;
}

static int
do_test (void)
{
  int result = 0;

  result |= test_except (0, "0");
  result |= test_except (FE_ALL_EXCEPT, "FE_ALL_EXCEPT");
#ifdef FE_DIVBYZERO
  result |= test_except (FE_DIVBYZERO, "FE_DIVBYZERO");
#endif
#ifdef FE_INEXACT
  result |= test_except (FE_INEXACT, "FE_INEXACT");
#endif
#ifdef FE_INVALID
  result |= test_except (FE_INVALID, "FE_INVALID");
#endif
#ifdef FE_OVERFLOW
  result |= test_except (FE_OVERFLOW, "FE_OVERFLOW");
#endif
#ifdef FE_UNDERFLOW
  result |= test_except (FE_UNDERFLOW, "FE_UNDERFLOW");
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
