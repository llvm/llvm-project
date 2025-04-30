/* Make sure pthread_mutex_timedlock doesn't return spurious error codes.

   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xsignal.h>
#include <support/xthread.h>
#include <support/xtime.h>
#include <time.h>
#include <unistd.h>

#define NANO_PER_SEC 1000000000LL
#define TIMEOUT (NANO_PER_SEC / 1000LL)
#define NUM_THREADS 50
#define RETEST_TIMES 100

static pthread_mutex_t mutex;
static int runs;
static clockid_t clockid;

static void
signal_handler (int sig_num)
{
  TEST_COMPARE (sig_num, SIGUSR1);
}

/* Call pthread_mutex_timedlock()/pthread_mutex_unlock() repetitively, hoping
   that one of them returns EAGAIN or EINTR unexpectedly.  */
static void *
worker_timedlock (void *arg)
{
  for (unsigned int run = 0; run < runs; run++)
    {
      struct timespec abs_time = timespec_add (xclock_now (CLOCK_REALTIME),
					       make_timespec (0, 1000000));

      int ret = pthread_mutex_timedlock (&mutex, &abs_time);

      if (ret == 0)
	xpthread_mutex_unlock (&mutex);

      TEST_VERIFY_EXIT (ret == 0 || ret == ETIMEDOUT);
    }
  return NULL;
}

static void *
worker_clocklock (void *arg)
{
  for (unsigned int run = 0; run < runs; run++)
    {
      struct timespec time =
	timespec_add (xclock_now (clockid), make_timespec (0, 1000000));

      int ret = pthread_mutex_clocklock (&mutex, clockid, &time);

      if (ret == 0)
	xpthread_mutex_unlock (&mutex);

      TEST_VERIFY_EXIT (ret == 0 || ret == ETIMEDOUT);
    }
  return NULL;
}

static int
run_test_set (void *(*worker) (void *))
{
  pthread_t workers[NUM_THREADS];

  /* Check if default pthread_mutex_{timed,clock}lock with valid arguments
     returns either 0 or ETIMEDOUT.  Since there is no easy way to force
     the error condition, the test creates multiple threads which in turn
     issues pthread_mutex_timedlock multiple times.  */
  runs = 100;
  for (int run = 0; run < RETEST_TIMES; run++)
    {
      for (int i = 0; i < NUM_THREADS; i++)
	workers[i] = xpthread_create (NULL, worker, NULL);
      for (int i = 0; i < NUM_THREADS; i++)
	xpthread_join (workers[i]);
    }

  /* The idea is similar to previous tests, but we check if
     pthread_mutex_{timed,clock}lock does not return EINTR.  */
  pthread_t thread;
  runs = 1;
  for (int i = 0; i < RETEST_TIMES * 1000; i++)
    {
      xpthread_mutex_lock (&mutex);
      thread = xpthread_create (NULL, worker, NULL);
      /* Sleep just a little bit to reach the lock on the worker thread.  */
      usleep (10);
      pthread_kill (thread, SIGUSR1);
      xpthread_mutex_unlock (&mutex);
      xpthread_join (thread);
    }

  return 0;
}

static int
do_test (void)
{

  xsignal (SIGUSR1, signal_handler);

  xpthread_mutex_init (&mutex, NULL);

  run_test_set (worker_timedlock);
  clockid = CLOCK_REALTIME;
  run_test_set (worker_clocklock);
  clockid = CLOCK_MONOTONIC;
  run_test_set (worker_clocklock);
  return 0;
}

#include <support/test-driver.c>
