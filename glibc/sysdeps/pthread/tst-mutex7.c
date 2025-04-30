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
#include <time.h>

/* This test is a template for other tests to use.  Other tests define
   the following macros to change the behaviour of the template test.
   The test is very simple, it configures N threads given the parameters
   below and then proceeds to go through mutex lock and unlock
   operations in each thread as described before for the thread
   function.  */
#ifndef TYPE
# define TYPE PTHREAD_MUTEX_DEFAULT
#endif
#ifndef ROBUST
# define ROBUST PTHREAD_MUTEX_STALLED
#endif
#ifndef DELAY_NSEC
# define DELAY_NSEC 11000
#endif
#ifndef ROUNDS
# define ROUNDS 1000
#endif
#ifndef N
# define N 100
#endif

static pthread_mutex_t lock;

/* Each thread locks and the subsequently unlocks the lock, yielding
   the smallest critical section possible.  After the unlock the thread
   waits DELAY_NSEC nanoseconds before doing the lock and unlock again.
   Every thread does this ROUNDS times.  The lock and unlock are
   checked for errors.  */
static void *
tf (void *arg)
{
  int nr = (long int) arg;
  int cnt;
  struct timespec ts = { .tv_sec = 0, .tv_nsec = DELAY_NSEC };

  for (cnt = 0; cnt < ROUNDS; ++cnt)
    {
      if (pthread_mutex_lock (&lock) != 0)
	{
	  printf ("thread %d: failed to get the lock\n", nr);
	  return (void *) 1l;
	}

      if (pthread_mutex_unlock (&lock) != 0)
	{
	  printf ("thread %d: failed to release the lock\n", nr);
	  return (void *) 1l;
	}

      if ((ts.tv_sec > 0) || (ts.tv_nsec > 0))
	nanosleep (&ts, NULL);
    }

  return NULL;
}

/* Setup and run N threads, where each thread does as described
   in the above thread function.  The threads are given a minimal 1MiB
   stack since they don't do anything between the lock and unlock.  */
static int
do_test (void)
{
  pthread_mutexattr_t a;

  if (pthread_mutexattr_init (&a) != 0)
    {
      puts ("mutexattr_init failed");
      exit (1);
    }

  if (pthread_mutexattr_settype (&a, TYPE) != 0)
    {
      puts ("mutexattr_settype failed");
      exit (1);
    }

  if (pthread_mutexattr_setrobust (&a, ROBUST) != 0)
    {
      puts ("mutexattr_setrobust failed");
      exit (1);
    }

#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif

  int e = pthread_mutex_init (&lock, &a);
  if (e != 0)
    {
#ifdef ENABLE_PI
      if (e == ENOTSUP)
	{
	  puts ("PI mutexes unsupported");
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

  pthread_attr_t at;
  pthread_t th[N];
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

  if (pthread_mutex_lock (&lock) != 0)
    {
      puts ("locking in parent failed");
      return 1;
    }

  for (cnt = 0; cnt < N; ++cnt)
    if (pthread_create (&th[cnt], &at, tf, (void *) (long int) cnt) != 0)
      {
	printf ("creating thread %d failed\n", cnt);
	return 1;
      }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  if (pthread_mutex_unlock (&lock) != 0)
    {
      puts ("unlocking in parent failed");
      return 1;
    }

  for (cnt = 0; cnt < N; ++cnt)
    if (pthread_join (th[cnt], NULL) != 0)
      {
	printf ("joining thread %d failed\n", cnt);
	return 1;
      }

  return 0;
}

#define TIMEOUT 60
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
