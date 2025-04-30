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

#include <error.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


#define N 10
#define ROUNDS 100

static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t bN1;
static pthread_barrier_t b2;


static void *
tf (void *p)
{
  if (pthread_mutex_lock (&mut) != 0)
    {
      puts ("child: 1st mutex_lock failed");
      exit (1);
    }

  int e = pthread_barrier_wait (&b2);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child: 1st barrier_wait failed");
      exit (1);
    }

  if (pthread_cond_wait (&cond, &mut) != 0)
    {
      puts ("child: cond_wait failed");
      exit (1);
    }

  if (pthread_mutex_unlock (&mut) != 0)
    {
      puts ("child: mutex_unlock failed");
      exit (1);
    }

  e = pthread_barrier_wait (&bN1);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child: 2nd barrier_wait failed");
      exit (1);
    }

  return NULL;
}


static int
do_test (void)
{
  if (pthread_barrier_init (&bN1, NULL, N + 1) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  if (pthread_barrier_init (&b2, NULL, 2) != 0)
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

  int r;
  for (r = 0; r < ROUNDS; ++r)
    {
      printf ("round %d\n", r + 1);

      int i;
      pthread_t th[N];
      for (i = 0; i < N; ++i)
	{
	  if (pthread_create (&th[i], &at, tf, NULL) != 0)
	    {
	      puts ("create failed");
	      exit (1);
	    }

	  int e = pthread_barrier_wait (&b2);
	  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	    {
	      puts ("parent: 1st barrier_wait failed");
	      exit (1);
	    }
	}

      if (pthread_mutex_lock (&mut) != 0)
	{
	  puts ("parent: mutex_lock failed");
	  exit (1);
	}
      if (pthread_mutex_unlock (&mut) != 0)
	{
	  puts ("parent: mutex_unlock failed");
	  exit (1);
	}

      /* N single signal calls.  Without locking.  This tests that no
	 signal gets lost.  */
      for (i = 0; i < N; ++i)
	if (pthread_cond_signal (&cond) != 0)
	  {
	    puts ("cond_signal failed");
	    exit (1);
	  }

      int e = pthread_barrier_wait (&bN1);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("parent: 2nd barrier_wait failed");
	  exit (1);
	}

      for (i = 0; i < N; ++i)
	if (pthread_join (th[i], NULL) != 0)
	  {
	    puts ("join failed");
	    exit (1);
	  }
    }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
