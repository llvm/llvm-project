/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  sigset_t ss;

  sigemptyset (&ss);

  int i;
  for (i = 0; i < 10000; ++i)
    {
      long int r = random ();

      if (r != SIG_BLOCK && r != SIG_SETMASK && r != SIG_UNBLOCK)
	{
	  int e = pthread_sigmask (r, &ss, NULL);

	  if (e == 0)
	    {
	      printf ("pthread_sigmask succeeded for how = %ld\n", r);
	      exit (1);
	    }

	  if (e != EINVAL)
	    {
	      puts ("pthread_sigmask didn't return EINVAL");
	      exit (1);
	    }
	}
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
