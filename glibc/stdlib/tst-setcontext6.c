/* Check getcontext and setcontext on the context from makecontext.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <ucontext.h>
#include <unistd.h>
#include <stdatomic.h>

static ucontext_t ctx[3];
static atomic_int done;

static void
f1 (void)
{
  printf ("start f1\n");
  if (!done)
    {
      if (getcontext (&ctx[2]) != 0)
	{
	  printf ("%s: getcontext: %m\n", __FUNCTION__);
	  exit (EXIT_FAILURE);
	}
      if (done)
	exit (EXIT_SUCCESS);
    }
  done++;
  if (setcontext (&ctx[2]) != 0)
    {
      printf ("%s: setcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
}

static int
do_test (void)
{
  char st1[32768];
  puts ("making contexts");
  if (getcontext (&ctx[0]) != 0)
    {
      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
  if (getcontext (&ctx[1]) != 0)
    {
      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
  ctx[1].uc_stack.ss_sp = st1;
  ctx[1].uc_stack.ss_size = sizeof st1;
  ctx[1].uc_link = &ctx[0];
  makecontext (&ctx[1], (void (*) (void)) f1, 0);
  if (setcontext (&ctx[1]) != 0)
    {
      printf ("%s: setcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
  exit (EXIT_FAILURE);
}

#include <support/test-driver.c>
