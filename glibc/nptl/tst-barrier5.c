/* This tests the barrier reset mechanism.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <internaltypes.h>


static pthread_barrier_t b1;
static pthread_barrier_t b2;


#define N 20
#define ROUNDS_PER_RUN 20
#define START ((BARRIER_IN_THRESHOLD / N - ROUNDS_PER_RUN / 2) * N)

static void *
tf (void *arg)
{
  int runs = 0;

  while (runs++ < 30)
    {
      /* In each run, we execute a number of rounds and initialize the barrier
	 so that we will go over the reset threshold with those rounds.  */
      for (int rounds = 0; rounds < ROUNDS_PER_RUN; rounds++)
	pthread_barrier_wait (&b1);

      if (pthread_barrier_wait (&b1) == PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  pthread_barrier_destroy (&b1);
	  if (pthread_barrier_init (&b1, NULL, N) != 0)
	    {
	      puts ("tf: 1st barrier_init failed");
	      exit (1);
	    }
	  puts ("b1 reinitialized");
	  /* Trigger a reset.  */
	  struct pthread_barrier *bar = (struct pthread_barrier *) &b1;
	  bar->in = START;
	  bar->out = START;
	  /* We deliberately don't set bar->current_round so that we also
	     test whether the helping for the updates of current_round
	     works correctly.  */
	}

      /* Same as above, just for b2.  */
      for (int rounds = 0; rounds < ROUNDS_PER_RUN; rounds++)
	pthread_barrier_wait (&b2);

      if (pthread_barrier_wait (&b2) == PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  pthread_barrier_destroy (&b2);
	  if (pthread_barrier_init (&b2, NULL, N) != 0)
	    {
	      puts ("tf: 2nd barrier_init failed");
	      exit (1);
	    }
	  puts ("b2 reinitialized");
	  /* Trigger a reset.  See above.  */
	  struct pthread_barrier *bar = (struct pthread_barrier *) &b2;
	  bar->in = START;
	  bar->out = START;
	}
    }

  return NULL;
}


static int
do_test (void)
{
  pthread_attr_t at;
  int cnt;

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

  if (pthread_barrier_init (&b1, NULL, N) != 0)
    {
      puts ("1st barrier_init failed");
      return 1;
    }

  if (pthread_barrier_init (&b2, NULL, N) != 0)
    {
      puts ("2nd barrier_init failed");
      return 1;
    }

  pthread_t th[N - 1];
  for (cnt = 0; cnt < N - 1; ++cnt)
    if (pthread_create (&th[cnt], &at, tf, NULL) != 0)
      {
	puts ("pthread_create failed");
	return 1;
      }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  tf (NULL);

  for (cnt = 0; cnt < N - 1; ++cnt)
    if (pthread_join (th[cnt], NULL) != 0)
      {
	puts ("pthread_join failed");
	return 1;
      }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
