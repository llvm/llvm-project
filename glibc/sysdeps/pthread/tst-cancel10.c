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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void
cleanup (void *arg)
{
  /* Just for fun.  */
  if (pthread_cancel (pthread_self ()) != 0)
    {
      puts ("cleanup: cancel failed");
      exit (1);
    }

  printf ("cleanup for %ld\n", (long int) arg);
}


static void *
tf (void *arg)
{
  long int n = (long int) arg;

  pthread_cleanup_push (cleanup, arg);

  if (pthread_setcanceltype ((n & 1) == 0
			     ? PTHREAD_CANCEL_DEFERRED
			     : PTHREAD_CANCEL_ASYNCHRONOUS, NULL) != 0)
    {
      puts ("setcanceltype failed");
      exit (1);
    }

  if (pthread_cancel (pthread_self ()) != 0)
    {
      puts ("cancel failed");
      exit (1);
    }

  pthread_testcancel ();

  /* We should never come here.  */

  pthread_cleanup_pop (0);

  return NULL;
}


static int
do_test (void)
{
  pthread_attr_t at;

  if (pthread_attr_init (&at) != 0)
    {
      puts ("attr_init failed");
      return 1;
    }

  if (pthread_attr_setstacksize (&at, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

#define N 20
  int i;
  pthread_t th[N];

  for (i = 0; i < N; ++i)
    if (pthread_create (&th[i], &at, tf, (void *) (long int) i) != 0)
      {
	puts ("create failed");
	exit (1);
      }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  for (i = 0; i < N; ++i)
    {
      void *r;
      if (pthread_join (th[i], &r) != 0)
	{
	  puts ("join failed");
	  exit (1);
	}

      if (r != PTHREAD_CANCELED)
	{
	  puts ("thread not canceled");
	  exit (1);
	}
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
