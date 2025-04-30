/* Test nearbyint functions do not clear exceptions (bug 15491).
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
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include <math-tests.h>

#ifndef FE_INVALID
# define FE_INVALID 0
#endif

static bool any_supported = false;

#define TEST_FUNC(NAME, FLOAT, SUFFIX)					\
static int								\
NAME (void)								\
{									\
  int result = 0;							\
  if (!EXCEPTION_TESTS (FLOAT))						\
    return 0;								\
  any_supported = true;							\
  volatile FLOAT a, b __attribute__ ((unused));				\
  a = 1.0;								\
  /* nearbyint must not clear already-raised exceptions.  */		\
  feraiseexcept (FE_ALL_EXCEPT);					\
  b = nearbyint ## SUFFIX (a);						\
  if (fetestexcept (FE_ALL_EXCEPT) == FE_ALL_EXCEPT)			\
    puts ("PASS: " #FLOAT);						\
  else									\
    {									\
      puts ("FAIL: " #FLOAT);						\
      result = 1;							\
    }									\
  /* But it mustn't lose exceptions from sNaN arguments.  */		\
  if (SNAN_TESTS (FLOAT))						\
    {									\
      static volatile FLOAT snan = __builtin_nans ## SUFFIX ("");	\
      volatile FLOAT c __attribute__ ((unused));			\
      feclearexcept (FE_ALL_EXCEPT);					\
      c = nearbyint ## SUFFIX (snan);					\
      if (fetestexcept (FE_INVALID) == FE_INVALID)			\
	puts ("PASS: " #FLOAT " sNaN");					\
      else								\
	{								\
	  puts ("FAIL: " #FLOAT " sNaN");				\
	  result = 1;							\
	}								\
    }									\
  return result;							\
}

TEST_FUNC (float_test, float, f)
TEST_FUNC (double_test, double, )
TEST_FUNC (ldouble_test, long double, l)

static int
do_test (void)
{
  int result = float_test ();
  result |= double_test ();
  result |= ldouble_test ();
  if (!any_supported)
    return 77;
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
