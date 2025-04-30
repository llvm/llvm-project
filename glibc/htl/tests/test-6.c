/* Test barriers.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#define _GNU_SOURCE

#include <pthread.h>
#include <stdio.h>
#include <error.h>
#include <assert.h>
#include <errno.h>

#define THREADS 500
#define WAITS 3

void *
dowait (void *arg)
{
  pthread_barrier_t *barrier = arg;
  int ret;

  ret = pthread_barrier_wait (barrier);
  printf ("%d ", pthread_self ());
  return (void *) ret;
}

int
main (int argc, char **argv)
{
  pthread_barrierattr_t attr;
  pthread_barrier_t barrier;

  int i, j;
  error_t err;
  pthread_t tid[THREADS];

  int havesyncs;

  err = pthread_barrierattr_init (&attr);
  if (err)
    error (1, err, "pthread_barrierattr_init");

  err = pthread_barrierattr_getpshared (&attr, &i);
  if (err)
    error (1, err, "pthread_barrierattr_getpshared");
  assert (i == PTHREAD_PROCESS_PRIVATE || i == PTHREAD_PROCESS_SHARED);

  err = pthread_barrierattr_setpshared (&attr, PTHREAD_PROCESS_PRIVATE);
  if (err)
    error (1, err, "pthread_barrierattr_setpshared");

  err = pthread_barrier_init (&barrier, &attr, THREADS + 1);
  if (err)
    error (1, err, "pthread_barrier_init");

  for (j = 0; j < WAITS; j++)
    {

      for (i = 0; i < THREADS; i++)
	{
	  err = pthread_create (&tid[i], 0, dowait, &barrier);
	  if (err)
	    error (1, err, "pthread_create (%d)", i);
	}

      printf ("Manager will now call pthread_barrier_wait.\n");

      havesyncs
	  = pthread_barrier_wait (&barrier) == PTHREAD_BARRIER_SERIAL_THREAD
	  ? 1 : 0;

      for (i = THREADS - 1; i >= 0; i--)
	{
	  void *ret;
	  err = pthread_join (tid[i], &ret);
	  if (err)
	    error (1, err, "pthread_join");

	  switch ((int) ret)
	    {
	    case 0:
	      break;

	    case PTHREAD_BARRIER_SERIAL_THREAD:
	      havesyncs++;
	      break;

	    default:
	      assert (!"Unknown value returned from pthread_barrier_wait.");
	      break;
	    }
	}

      printf ("\n");

      assert (havesyncs == 1);
    }

  return 0;
}
