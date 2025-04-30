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

#include <error.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static pthread_barrier_t bar;


static void *
tf (void *a)
{
  int i = (long int) a;
  int err;

  printf ("child %d: lock\n", i);

  err = pthread_mutex_lock (&mut);
  if (err != 0)
    error (EXIT_FAILURE, err, "locking in child failed");

  printf ("child %d: sync\n", i);

  int e = pthread_barrier_wait (&bar);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child: barrier_wait failed");
      exit (1);
    }

  printf ("child %d: wait\n", i);

  err = pthread_cond_wait (&cond, &mut);
  if (err != 0)
    error (EXIT_FAILURE, err, "child %d: failed to wait", i);

  printf ("child %d: woken up\n", i);

  err = pthread_mutex_unlock (&mut);
  if (err != 0)
    error (EXIT_FAILURE, err, "child %d: unlock[2] failed", i);

  printf ("child %d: done\n", i);

  return NULL;
}


#define N 10


static int
do_test (void)
{
  pthread_t th[N];
  int i;
  int err;

  printf ("&cond = %p\n&mut = %p\n", &cond, &mut);

  if (pthread_barrier_init (&bar, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

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

  for (i = 0; i < N; ++i)
    {
      printf ("create thread %d\n", i);

      err = pthread_create (&th[i], &at, tf, (void *) (long int) i);
      if (err != 0)
	error (EXIT_FAILURE, err, "cannot create thread %d", i);

      printf ("wait for child %d\n", i);

      /* Wait for the child to start up and get the mutex for the
	 conditional variable.  */
      int e = pthread_barrier_wait (&bar);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait failed");
	  exit (1);
	}
    }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  puts ("get lock outselves");

  err = pthread_mutex_lock (&mut);
  if (err != 0)
    error (EXIT_FAILURE, err, "mut locking failed");

  puts ("broadcast");

  /* Wake up all threads.  */
  err = pthread_cond_broadcast (&cond);
  if (err != 0)
    error (EXIT_FAILURE, err, "parent: broadcast failed");

  err = pthread_mutex_unlock (&mut);
  if (err != 0)
    error (EXIT_FAILURE, err, "mut unlocking failed");

  /* Join all threads.  */
  for (i = 0; i < N; ++i)
    {
      printf ("join thread %d\n", i);

      err = pthread_join (th[i], NULL);
      if (err != 0)
	error (EXIT_FAILURE, err, "join of child %d failed", i);
    }

  puts ("done");

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
