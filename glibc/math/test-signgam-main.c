/* Test lgamma functions do not set signgam for ISO C.
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

#undef _LIBC
#undef _GNU_SOURCE

#include <math.h>
#include <stdio.h>

#define INITVAL ((TYPE) -1 / 3)

#if DO_INIT
TYPE signgam = INITVAL;
#else
TYPE signgam;
#endif

#define RUN_TESTS(FUNC, TYPE)					\
  do								\
    {								\
      volatile TYPE a, b, c __attribute__ ((unused));		\
      a = 0.5;							\
      b = -0.5;							\
      signgam = INITVAL;					\
      c = FUNC (a);						\
      if (signgam == INITVAL)					\
	puts ("PASS: " #FUNC " (0.5) setting signgam");		\
      else							\
	{							\
	  puts ("FAIL: " #FUNC " (0.5) setting signgam");	\
	  result = 1;						\
	}							\
      signgam = INITVAL;					\
      c = FUNC (b);						\
      if (signgam == INITVAL)					\
	puts ("PASS: " #FUNC " (-0.5) setting signgam");	\
      else							\
	{							\
	  puts ("FAIL: " #FUNC " (-0.5) setting signgam");	\
	  result = 1;						\
	}							\
    }								\
  while (0)

int
main (void)
{
  int result = 0;
  RUN_TESTS (lgammaf, float);
  RUN_TESTS (lgamma, double);
  RUN_TESTS (lgammal, long double);
  return result;
}
