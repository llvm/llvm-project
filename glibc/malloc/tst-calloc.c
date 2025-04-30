/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>.

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

#include <errno.h>
#include <error.h>
#include <limits.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <libc-diag.h>


/* Number of samples per size.  */
#define N 50000


static void
fixed_test (int size)
{
  char *ptrs[N];
  int i;

  for (i = 0; i < N; ++i)
    {
      int j;

      ptrs[i] = (char *) calloc (1, size);

      if (ptrs[i] == NULL)
	break;

      for (j = 0; j < size; ++j)
	{
	  if (ptrs[i][j] != '\0')
	    error (EXIT_FAILURE, 0,
		   "byte not cleared (size %d, element %d, byte %d)",
		   size, i, j);
	  ptrs[i][j] = '\xff';
	}
    }

  while (i-- > 0)
    free (ptrs[i]);
}


static void
random_test (void)
{
  char *ptrs[N];
  int i;

  for (i = 0; i < N; ++i)
    {
      int j;
      int n = 1 + random () % 10;
      int elem = 1 + random () % 100;
      int size = n * elem;

      ptrs[i] = (char *) calloc (n, elem);

      if (ptrs[i] == NULL)
	break;

      for (j = 0; j < size; ++j)
	{
	  if (ptrs[i][j] != '\0')
	    error (EXIT_FAILURE, 0,
		   "byte not cleared (size %d, element %d, byte %d)",
		   size, i, j);
	  ptrs[i][j] = '\xff';
	}
    }

  while (i-- > 0)
    free (ptrs[i]);
}


static void
null_test (void)
{
  /* If the size is 0 the result is implementation defined.  Just make
     sure the program doesn't crash.  The result of calloc is
     deliberately ignored, so do not warn about that.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (10, "-Wunused-result");
  calloc (0, 0);
  calloc (0, UINT_MAX);
  calloc (UINT_MAX, 0);
  calloc (0, ~((size_t) 0));
  calloc (~((size_t) 0), 0);
  DIAG_POP_NEEDS_COMMENT;
}


static int
do_test (void)
{
  /* We are allocating blocks with `calloc' and check whether every
     block is completely cleared.  We first try this for some fixed
     times and then with random size.  */
  fixed_test (15);
  fixed_test (5);
  fixed_test (17);
  fixed_test (6);
  fixed_test (31);
  fixed_test (96);

  random_test ();

  null_test ();

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
