/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

/* Test case for Bugzilla # 269 */

#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>

jmp_buf buf1;
jmp_buf buf2;
int *p;
int n_x = 6;

static int g_counter = 0;

int
f (void)
{
  static int counter = 0;
  static int way_point1 = 3;
  static int way_point2 = 2;
  int lose = 0;

  if (setjmp (buf1) != 101)
    {
      int a[n_x];		/* reallocate stack space */
      g_counter++;
      p = &a[0];
      if (g_counter < 5)
	longjmp (buf1, 2);
      else if (g_counter == 5)
	longjmp (buf1, 101);
      else
	{
	  _setjmp (buf2);
	  _longjmp (buf1, 101);
	}
    }

  way_point1--;

  if (counter == 0)
    {
      counter++;
      {
	int a[n_x];		/* reallocate stack space */
	g_counter++;
	p = &a[0];
	if (g_counter < 5)
	  longjmp (buf1, 2);
	else if (g_counter == 5)
	  longjmp (buf1, 101);
	else
	  {
	    _setjmp (buf2);
	    _longjmp (buf1, 101);
	  }
      }
    }

  way_point2--;

  if (counter == 1)
    {
      counter++;
      longjmp (buf2, 2);
    }

  lose = !(way_point1 == 0 && way_point2 == 0
	   && g_counter == 6 && counter == 2);

  return lose;
}

static int
do_test (void)
{
  int lose;

  lose = f ();

  if (lose)
    puts ("Test FAILED!");
  else
    puts ("Test succeeded!");

  return lose ? EXIT_FAILURE : EXIT_SUCCESS;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
