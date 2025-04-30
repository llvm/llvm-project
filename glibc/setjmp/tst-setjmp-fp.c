/* Test that setjmp/longjmp do not save and restore floating-point
   exceptions and rounding modes.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>

static jmp_buf __attribute__ ((unused)) env;
static int result = 0;

#if defined FE_TONEAREST && defined FE_TOWARDZERO
static int expected_rounding_mode = FE_TONEAREST;

static void
change_rounding_mode (void)
{
  if (fesetround (FE_TOWARDZERO) == 0)
    expected_rounding_mode = FE_TOWARDZERO;
  else
    puts ("fesetround (FE_TOWARDZERO) failed, continuing test");
  longjmp (env, 1);
}
#endif

#ifdef FE_INVALID
static int expected_exceptions = 0;

static void
raise_exception (void)
{
  if (feraiseexcept (FE_INVALID) == 0)
    expected_exceptions = FE_INVALID;
  else
    puts ("feraiseexcept (FE_INVALID) failed, continuing test");
  longjmp (env, 1);
}
#endif

static int
do_test (void)
{
#if defined FE_TONEAREST && defined FE_TOWARDZERO
  if (fesetround (FE_TONEAREST) == 0)
    {
      if (setjmp (env) == 0)
	change_rounding_mode ();
      else
	{
	  if (fegetround () == expected_rounding_mode)
	    puts ("PASS: longjmp preserved rounding mode");
	  else
	    {
	      puts ("FAIL: longjmp changed rounding mode");
	      result = 1;
	    }
	}
    }
  else
    puts ("fesetround (FE_TONEAREST) failed, not testing rounding modes");
#else
  puts ("rounding mode test not supported");
#endif
#ifdef FE_INVALID
  if (feclearexcept (FE_ALL_EXCEPT) == 0)
    {
      if (setjmp (env) == 0)
	raise_exception ();
      else
	{
	  if (fetestexcept (FE_INVALID) == expected_exceptions)
	    puts ("PASS: longjmp preserved exceptions");
	  else
	    {
	      puts ("FAIL: longjmp changed exceptions");
	      result = 1;
	    }
	}
    }
  else
    puts ("feclearexcept (FE_ALL_EXCEPT) failed, not testing exceptions");
#else
  puts ("exception test not supported");
#endif
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
