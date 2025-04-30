/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2005.

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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


static pthread_barrier_t b;
static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t m;
static bool first = true;


static void *
tf (void *arg)
{
  long int n = (long int) arg;

  if (pthread_mutex_lock (&m) != 0)
    {
      printf ("thread %ld: mutex_lock failed\n", n + 1);
      exit (1);
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf ("thread %ld: barrier_wait failed\n", n + 1);
      exit (1);
    }

  e = pthread_cond_wait (&c, &m);
  if (first)
    {
      if (e != 0)
	{
	  printf ("thread %ld: cond_wait failed\n", n + 1);
	  exit (1);
	}
      first = false;
    }
  else
    {
      if (e != EOWNERDEAD)
	{
	  printf ("thread %ld: cond_wait did not return EOWNERDEAD\n", n + 1);
	  exit (1);
	}
    }

  if (pthread_cancel (pthread_self ()) != 0)
    {
      printf ("thread %ld: cancel failed\n", n + 1);
      exit (1);
    }

  pthread_testcancel ();

  printf ("thread %ld: testcancel returned\n", n + 1);
  exit (1);
}


static int
do_test (void)
{
  pthread_mutexattr_t a;
  if (pthread_mutexattr_init (&a) != 0)
    {
      puts ("mutexattr_init failed");
      return 1;
    }

  if (pthread_mutexattr_setrobust (&a, PTHREAD_MUTEX_ROBUST_NP) != 0)
    {
      puts ("mutexattr_setrobust failed");
      return 1;
    }

#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif

  int e;
  e = pthread_mutex_init (&m, &a);
  if (e != 0)
    {
#ifdef ENABLE_PI
      if (e == ENOTSUP)
	{
	  puts ("PI robust mutexes not supported");
	  return 0;
	}
#endif
      puts ("mutex_init failed");
      return 1;
    }

  if (pthread_mutexattr_destroy (&a) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

#define N 5
  pthread_t th[N];
  for (long int n = 0; n < N; ++n)
    {
      if (pthread_create (&th[n], NULL, tf, (void *) n) != 0)
	{
	  printf ("pthread_create loop %ld failed\n", n + 1);
	  return 1;
	}

      e = pthread_barrier_wait (&b);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("parent: barrier_wait failed in round %ld\n", n + 1);
	  return 1;
	}
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("parent: mutex_lock failed");
      return 1;
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("parent: mutex_unlock failed");
      return 1;
    }

  if (pthread_cond_broadcast (&c) != 0)
    {
      puts ("cond_broadcast failed");
      return 1;
    }

  for (int n = 0; n < N; ++n)
    {
      void *res;
      if (pthread_join (th[n], &res) != 0)
	{
	  printf ("join round %d failed\n", n + 1);
	  return 1;
	}
      if (res != PTHREAD_CANCELED)
	{
	  printf ("thread %d not canceled\n", n + 1);
	  return 1;
	}
    }

  e = pthread_mutex_lock (&m);
  if (e == 0)
    {
      puts ("parent: 2nd mutex_lock succeeded");
      return 1;
    }
  if (e != EOWNERDEAD)
    {
      puts ("parent: mutex_lock did not return EOWNERDEAD");
      return 1;
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("parent: 2nd mutex_unlock failed");
      return 1;
    }

  if (pthread_mutex_destroy (&m) != 0)
    {
      puts ("mutex_destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
