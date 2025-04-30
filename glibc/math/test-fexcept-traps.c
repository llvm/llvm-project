/* Test fegetexceptflag and fesetexceptflag: exception traps enabled.
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
do_test (void)
{
  int result = 0;

  fedisableexcept (FE_ALL_EXCEPT);
  int ret = feraiseexcept (FE_ALL_EXCEPT);
  if (ret != 0)
    {
      if (EXCEPTION_TESTS (float))
	{
	  puts ("feraiseexcept (FE_ALL_EXCEPT) failed");
	  result = 1;
	  return result;
	}
      else
	{
	  puts ("feraiseexcept (FE_ALL_EXCEPT) unsupported, cannot test");
	  return 77;
	}
    }
  fexcept_t saved;
  ret = fegetexceptflag (&saved, FE_ALL_EXCEPT);
  if (ret != 0)
    {
      puts ("fegetexceptflag failed");
      result = 1;
      return result;
    }
  feclearexcept (FE_ALL_EXCEPT);

  ret = feenableexcept (FE_ALL_EXCEPT);
  if (!EXCEPTION_ENABLE_SUPPORTED (FE_ALL_EXCEPT) && (ret == -1))
    {
      puts ("feenableexcept (FE_ALL_EXCEPT) not supported, cannot test");
      return 77;
    }
  else if (ret != 0)
    {
      puts ("feenableexcept (FE_ALL_EXCEPT) failed");
      result = 1;
    }

  if (EXCEPTION_SET_FORCES_TRAP)
    {
      puts ("setting exceptions traps, cannot test on this architecture");
      return 77;
    }
  /* The test is that this does not cause exception traps.  */
  ret = fesetexceptflag (&saved, FE_ALL_EXCEPT);
  if (ret != 0)
    {
      puts ("fesetexceptflag failed");
      result = 1;
    }
  feclearexcept (FE_ALL_EXCEPT);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
