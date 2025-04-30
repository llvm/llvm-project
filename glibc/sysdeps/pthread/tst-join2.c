/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


static void *
tf (void *arg)
{
  if (pthread_mutex_lock (&lock) != 0)
    {
      puts ("child: mutex_lock failed");
      return NULL;
    }

  return (void *) 42l;
}


static int
do_test (void)
{
  pthread_t th;

  if (pthread_mutex_lock (&lock) != 0)
    {
      puts ("mutex_lock failed");
      exit (1);
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("mutex_create failed");
      exit (1);
    }

  void *status;
  int val = pthread_tryjoin_np (th, &status);
  if (val == 0)
    {
      puts ("1st tryjoin succeeded");
      exit (1);
    }
  else if (val != EBUSY)
    {
      puts ("1st tryjoin didn't return EBUSY");
      exit (1);
    }

  if (pthread_mutex_unlock (&lock) != 0)
    {
      puts ("mutex_unlock failed");
      exit (1);
    }

  while ((val = pthread_tryjoin_np (th, &status)) != 0)
    {
      if (val != EBUSY)
	{
	  printf ("tryjoin returned %s (%d), expected only 0 or EBUSY\n",
		  strerror (val), val);
	  exit (1);
	}

      /* Delay minimally.  */
      struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000 };
      nanosleep (&ts, NULL);
    }

  if (status != (void *) 42l)
    {
      printf ("return value %p, expected %p\n", status, (void *) 42l);
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
