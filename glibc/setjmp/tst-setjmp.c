/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>

static jmp_buf env;
static int last_value = -1, lose = 0;

static __attribute__ ((__noreturn__)) void
jump (int val)
{
  longjmp (env, val);
}

static int
do_test (void)
{
  int value;

  value = setjmp (env);
  if (value != last_value + 1)
    {
      fputs("Shouldn't have ", stdout);
      lose = 1;
    }
  last_value = value;
  switch (value)
    {
    case 0:
      puts("Saved environment.");
      jump (0);
    default:
      printf ("Jumped to %d.\n", value);
      if (value < 10)
	jump (value + 1);
    }

  if (!lose && value == 10)
    {
      /* Do a second test, this time without `setjmp' being a macro.
         This is not required by ISO C but we have this for compatibility.  */
#undef setjmp
      extern int setjmp (jmp_buf);

      last_value = -1;
      lose = 0;

      value = setjmp (env);
      if (value != last_value + 1)
	{
	  fputs("Shouldn't have ", stdout);
	  lose = 1;
	}
      last_value = value;
      switch (value)
	{
	case 0:
	  puts("Saved environment.");
	  jump (0);
	default:
	  printf ("Jumped to %d.\n", value);
	  if (value < 10)
	    jump (value + 1);
	}
    }

  if (!lose && value == 10)
    {
      /* And again for the `_setjmp' function.  */
#ifndef _setjmp
      extern int _setjmp (jmp_buf);
#endif
      last_value = -1;
      lose = 0;

      value = _setjmp (env);
      if (value != last_value + 1)
	{
	  fputs("Shouldn't have ", stdout);
	  lose = 1;
	}
      last_value = value;
      switch (value)
	{
	case 0:
	  puts("Saved environment.");
	  jump (0);
	default:
	  printf ("Jumped to %d.\n", value);
	  if (value < 10)
	    jump (value + 1);
	}
    }

  if (lose || value != 10)
    puts ("Test FAILED!");
  else
    puts ("Test succeeded!");

  return lose ? EXIT_FAILURE : EXIT_SUCCESS;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
