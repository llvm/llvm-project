/* Test iszero with excess precision.
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

#include <float.h>
#include <math.h>
#include <stdio.h>

#define TEST(TYPE, TRUE_MIN)						\
  do									\
    {									\
      if (iszero (TRUE_MIN / 2))					\
	puts ("iszero removes excess precision for " #TYPE);		\
      else								\
	{								\
	  puts ("iszero fails to remove excess precision for " #TYPE);	\
	  result = 1;							\
	}								\
    }									\
  while (0)

static int
do_test (void)
{
  int result = 0;

  TEST (float, FLT_TRUE_MIN);
  TEST (double, DBL_TRUE_MIN);
  TEST (long double, LDBL_TRUE_MIN);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
