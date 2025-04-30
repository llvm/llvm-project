/* Test femode_t functions.
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
test_mmee (int mode1, int mode2, int exc1, int exc2)
{
  int result = 0;
  printf ("testing %x %x %x %x\n", (unsigned int) mode1, (unsigned int) mode2,
	  (unsigned int) exc1, (unsigned int) exc2);

  feclearexcept (FE_ALL_EXCEPT);
  int ret = fesetround (mode1);
  if (ret != 0)
    {
      if (ROUNDING_TESTS (float, mode1))
	{
	  puts ("first fesetround failed unexpectedly");
	  result = 1;
	}
      else
	puts ("first fesetround failed, cannot test");
      return result;
    }
  ret = fesetexcept (exc1);
  if (ret != 0)
    {
      if (EXCEPTION_TESTS (float) || exc1 == 0)
	{
	  puts ("first fesetexcept failed unexpectedly");
	  result = 1;
	}
      else
	puts ("first fesetexcept failed, cannot test");
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
  feclearexcept (FE_ALL_EXCEPT);
  ret = fesetround (mode2);
  if (ret != 0)
    {
      if (ROUNDING_TESTS (float, mode2))
	{
	  puts ("second fesetround failed unexpectedly");
	  result = 1;
	}
      else
	puts ("second fesetround failed, cannot test");
      return result;
    }
  ret = fesetexcept (exc2);
  if (ret != 0)
    {
      if (EXCEPTION_TESTS (float) || exc2 == 0)
	{
	  puts ("second fesetexcept failed unexpectedly");
	  result = 1;
	}
      else
	puts ("second fesetexcept failed, cannot test");
      return result;
    }
  ret = fesetmode (&saved);
  if (ret != 0)
    {
      puts ("fesetmode failed");
      result = 1;
      return result;
    }
  /* Verify that the rounding mode was restored but the exception
     flags remain unchanged.  */
  ret = fegetround ();
  if (ret != mode1)
    {
      printf ("restored rounding mode %x not %x\n", (unsigned int) ret,
	      (unsigned int) mode1);
      result = 1;
    }
  ret = fetestexcept (FE_ALL_EXCEPT);
  if (ret != exc2)
    {
      printf ("exceptions %x not %x\n", (unsigned int) ret,
	      (unsigned int) exc2);
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
  ret = fegetround ();
  if (ret != FE_TONEAREST)
    {
      printf ("FE_DFL_MODE rounding mode %x not %x\n", (unsigned int) ret,
	      (unsigned int) FE_TONEAREST);
      result = 1;
    }
  ret = fetestexcept (FE_ALL_EXCEPT);
  if (ret != exc2)
    {
      printf ("FE_DFL_MODE exceptions %x not %x\n", (unsigned int) ret,
	      (unsigned int) exc2);
      result = 1;
    }
  return result;
}

static int
test_mme (int mode1, int mode2, int exc1)
{
  int result = 0;

  result |= test_mmee (mode1, mode2, exc1, 0);
  result |= test_mmee (mode1, mode2, exc1, FE_ALL_EXCEPT);
#ifdef FE_DIVBYZERO
  result |= test_mmee (mode1, mode2, exc1, FE_DIVBYZERO);
#endif
#ifdef FE_INEXACT
  result |= test_mmee (mode1, mode2, exc1, FE_INEXACT);
#endif
#ifdef FE_INVALID
  result |= test_mmee (mode1, mode2, exc1, FE_INVALID);
#endif
#ifdef FE_OVERFLOW
  result |= test_mmee (mode1, mode2, exc1, FE_OVERFLOW);
#endif
#ifdef FE_UNDERFLOW
  result |= test_mmee (mode1, mode2, exc1, FE_UNDERFLOW);
#endif

  return result;
}

static int
test_mm (int mode1, int mode2)
{
  int result = 0;

  result |= test_mme (mode1, mode2, 0);
  result |= test_mme (mode1, mode2, FE_ALL_EXCEPT);
#ifdef FE_DIVBYZERO
  result |= test_mme (mode1, mode2, FE_DIVBYZERO);
#endif
#ifdef FE_INEXACT
  result |= test_mme (mode1, mode2, FE_INEXACT);
#endif
#ifdef FE_INVALID
  result |= test_mme (mode1, mode2, FE_INVALID);
#endif
#ifdef FE_OVERFLOW
  result |= test_mme (mode1, mode2, FE_OVERFLOW);
#endif
#ifdef FE_UNDERFLOW
  result |= test_mme (mode1, mode2, FE_UNDERFLOW);
#endif

  return result;
}

static int
test_m (int mode1)
{
  int result = 0;

#ifdef FE_DOWNWARD
  result |= test_mm (mode1, FE_DOWNWARD);
#endif
#ifdef FE_TONEAREST
  result |= test_mm (mode1, FE_TONEAREST);
#endif
#ifdef FE_TOWARDZERO
  result |= test_mm (mode1, FE_TOWARDZERO);
#endif
#ifdef FE_UPWARD
  result |= test_mm (mode1, FE_UPWARD);
#endif

  return result;
}

static int
do_test (void)
{
  int result = 0;

#ifdef FE_DOWNWARD
  result |= test_m (FE_DOWNWARD);
#endif
#ifdef FE_TONEAREST
  result |= test_m (FE_TONEAREST);
#endif
#ifdef FE_TOWARDZERO
  result |= test_m (FE_TOWARDZERO);
#endif
#ifdef FE_UPWARD
  result |= test_m (FE_UPWARD);
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
