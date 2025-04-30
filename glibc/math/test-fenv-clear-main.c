/* Test fesetenv (FE_DFL_ENV) and fesetenv (FE_NOMASK_ENV) clear
   exceptions (bug 19181).
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

#include <fenv.h>
#include <float.h>
#include <stdio.h>

volatile float fa = 1.0f, fb = 0.0f, fc = FLT_MAX, fr;
volatile long double lda = 1.0L, ldb = 0.0L, ldc = LDBL_MAX, ldr;

static void
raise_exceptions (void)
{
  /* Raise exceptions both with feraiseexcept and arithmetic to allow
     for case of multiple floating-point units with separate
     exceptions state.  */
  feraiseexcept (FE_ALL_EXCEPT);
  fr = fb / fb;
  fr = fa / fb;
  fr = fc * fc;
  fr = fa / fc / fc;
  ldr = ldb / ldb;
  ldr = lda / ldb;
  ldr = ldc * ldc;
  ldr = lda / ldc / ldc;
}

static __attribute__ ((noinline)) int
run_tests (void)
{
  int result = 0;
  raise_exceptions ();
  if (fesetenv (FE_DFL_ENV) == 0)
    {
      puts ("PASS: fesetenv (FE_DFL_ENV)");
      if (fetestexcept (FE_ALL_EXCEPT) == 0)
	puts ("PASS: fesetenv (FE_DFL_ENV) clearing exceptions");
      else
	{
	  puts ("FAIL: fesetenv (FE_DFL_ENV) clearing exceptions");
	  result = 1;
	}
    }
  else
    {
      puts ("FAIL: fesetenv (FE_DFL_ENV)");
      result = 1;
    }
#ifdef FE_NOMASK_ENV
  raise_exceptions ();
  if (fesetenv (FE_NOMASK_ENV) == 0)
    {
      if (fetestexcept (FE_ALL_EXCEPT) == 0)
	puts ("PASS: fesetenv (FE_NOMASK_ENV) clearing exceptions");
      else
	{
	  puts ("FAIL: fesetenv (FE_NOMASK_ENV) clearing exceptions");
	  result = 1;
	}
    }
  else
    puts ("fesetenv (FE_NOMASK_ENV) failed, cannot test");
#endif
  return result;
}

static int
do_test (void)
{
  CHECK_CAN_TEST;
  return run_tests ();
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
