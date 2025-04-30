/* Test initstate saving the old state.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2005.

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
#include <stdlib.h>

static int
do_test (void)
{
  int pass;
  int ret = 0;
  long int r[2];

  for (pass = 0; pass < 2; pass++)
    {
      srandom (0x12344321);

      int j;
      for (j = 0; j < 3; ++j)
	random ();
      if (pass == 1)
	{
	  char state[128];
	  char *ostate = initstate (0x34562101, state, 128);
	  if (setstate (ostate) != state)
	    {
	      puts ("setstate (ostate) != state");
	      ret = 1;
	    }
	}

      random ();
      r[pass] = random ();
    }

  if (r[0] != r[1])
    {
      printf ("%ld != %ld\n", r[0], r[1]);
      ret = 1;
    }
  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
