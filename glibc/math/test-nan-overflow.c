/* Test nan functions stack overflow (bug 16962).
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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <stdlib.h>

#define STACK_LIM 1048576
#define STRING_SIZE (2 * STACK_LIM)

static int
do_test (void)
{
  int result = 0;
  struct rlimit lim;
  getrlimit (RLIMIT_STACK, &lim);
  lim.rlim_cur = STACK_LIM;
  setrlimit (RLIMIT_STACK, &lim);
  char *nanstr = malloc (STRING_SIZE);
  if (nanstr == NULL)
    {
      puts ("malloc failed, cannot test");
      return 77;
    }
  memset (nanstr, '0', STRING_SIZE - 1);
  nanstr[STRING_SIZE - 1] = 0;
#define NAN_TEST(TYPE, FUNC)			\
  do						\
    {						\
      char *volatile p = nanstr;		\
      volatile TYPE v = FUNC (p);		\
      if (isnan (v))				\
	puts ("PASS: " #FUNC);			\
      else					\
	{					\
	  puts ("FAIL: " #FUNC);		\
	  result = 1;				\
	}					\
    }						\
  while (0)
  NAN_TEST (float, nanf);
  NAN_TEST (double, nan);
  NAN_TEST (long double, nanl);
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
