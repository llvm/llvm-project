/* Test return value when setting FE_NOMASK_ENV (BZ16918, BZ17009).
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

static int count_errors;

static void
test_feenableexcept (void)
{
#if defined FE_ALL_EXCEPT
  int res;

  fedisableexcept (FE_ALL_EXCEPT);

  res = feenableexcept (FE_ALL_EXCEPT);

  if (!EXCEPTION_ENABLE_SUPPORTED (FE_ALL_EXCEPT) && (res == -1))
    {
      puts ("feenableexcept (FE_ALL_EXCEPT) not supported, cannot test.");
      return;
    }
  else if (res != 0)
    {
      puts ("feenableexcept (FE_ALL_EXCEPT) failed");
      count_errors++;
    }

  if (fegetexcept () != FE_ALL_EXCEPT)
    {
      puts ("feenableexcept did not set all exceptions");
      count_errors++;
    }
#endif
}

static void
test_fesetenv (void)
{
#if defined FE_NOMASK_ENV && defined FE_ALL_EXCEPT
  int res;

  fedisableexcept (FE_ALL_EXCEPT);

  res = fesetenv (FE_NOMASK_ENV);

  if (!EXCEPTION_ENABLE_SUPPORTED (FE_ALL_EXCEPT) && (res != 0))
    {
      puts ("fesetenv (FE_NOMASK_ENV) not supported, cannot test.");
      return;
    }
  else if (res != 0)
    {
      puts ("fesetenv (FE_NOMASK_ENV) failed");
      count_errors++;
    }

  if (fegetexcept () != FE_ALL_EXCEPT)
    {
      puts ("fesetenv did not set all exceptions");
      count_errors++;
    }
#endif
}

static void
test_feupdateenv (void)
{
#if defined FE_NOMASK_ENV && defined FE_ALL_EXCEPT
  int res;

  fedisableexcept (FE_ALL_EXCEPT);

  res = feupdateenv (FE_NOMASK_ENV);

  if (!EXCEPTION_ENABLE_SUPPORTED (FE_ALL_EXCEPT) && (res != 0))
    {
      puts ("feupdateenv (FE_NOMASK_ENV)) not supported, cannot test.");
      return;
    }
  else if (res != 0)
    {
      puts ("feupdateenv (FE_NOMASK_ENV) failed");
      count_errors++;
    }

  if (fegetexcept () != FE_ALL_EXCEPT)
    {
      puts ("feupdateenv did not set all exceptions");
      count_errors++;
    }
#endif
}

static int
do_test (void)
{
  test_feenableexcept ();
  test_fesetenv ();
  test_feupdateenv ();

  return count_errors != 0 ? 1 : 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
