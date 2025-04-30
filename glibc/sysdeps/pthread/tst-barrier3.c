/* Test of POSIX barriers.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NTHREADS 20

#define ROUNDS 20

static pthread_barrier_t barriers[NTHREADS];

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int counters[NTHREADS];
static int serial[NTHREADS];

static void *
worker (void *arg)
{
  void *result = NULL;
  int nr = (long int) arg;
  int i;

  for (i = 0; i < ROUNDS; ++i)
    {
      int j;
      int retval;

      if (nr == 0)
	{
	  memset (counters, '\0', sizeof (counters));
	  memset (serial, '\0', sizeof (serial));
	}

      retval = pthread_barrier_wait (&barriers[NTHREADS - 1]);
      if (retval != 0 && retval != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("thread %d failed to wait for all the others\n", nr);
	  result = (void *) 1;
	}

      for (j = nr; j < NTHREADS; ++j)
	{
	  /* Increment the counter for this round.  */
	  pthread_mutex_lock (&lock);
	  ++counters[j];
	  pthread_mutex_unlock (&lock);

	  /* Wait for the rest.  */
	  retval = pthread_barrier_wait (&barriers[j]);

	  /* Test the result.  */
	  if (nr == 0 && counters[j] != j + 1)
	    {
	      printf ("barrier in round %d released but count is %d\n",
		      j, counters[j]);
	      result = (void *) 1;
	    }

	  if (retval != 0)
	    {
	      if (retval != PTHREAD_BARRIER_SERIAL_THREAD)
		{
		  printf ("thread %d in round %d has nonzero return value != PTHREAD_BARRIER_SERIAL_THREAD\n",
			  nr, j);
		  result = (void *) 1;
		}
	      else
		{
		  pthread_mutex_lock (&lock);
		  ++serial[j];
		  pthread_mutex_unlock (&lock);
		}
	    }

	  /* Wait for the rest again.  */
	  retval = pthread_barrier_wait (&barriers[j]);

	  /* Now we can check whether exactly one thread was serializing.  */
	  if (nr == 0 && serial[j] != 1)
	    {
	      printf ("not exactly one serial thread in round %d\n", j);
	      result = (void *) 1;
	    }
	}
    }

  return result;
}


#define TEST_FUNCTION do_test ()
#define TIMEOUT 60
static int
do_test (void)
{
  pthread_t threads[NTHREADS];
  int i;
  void *res;
  int result = 0;

  /* Initialized the barrier variables.  */
  for (i = 0; i < NTHREADS; ++i)
    if (pthread_barrier_init (&barriers[i], NULL, i + 1) != 0)
      {
	printf ("Failed to initialize barrier %d\n", i);
	exit (1);
      }

  /* Start the threads.  */
  for (i = 0; i < NTHREADS; ++i)
    if (pthread_create (&threads[i], NULL, worker, (void *) (long int) i) != 0)
      {
	printf ("Failed to start thread %d\n", i);
	exit (1);
      }

  /* And wait for them.  */
  for (i = 0; i < NTHREADS; ++i)
    if (pthread_join (threads[i], &res) != 0 || res != NULL)
      {
	printf ("thread %d returned a failure\n", i);
	result = 1;
      }
    else
      printf ("joined threads %d\n", i);

  if (result == 0)
    puts ("all OK");

  return result;
}

#include "../test-skeleton.c"
