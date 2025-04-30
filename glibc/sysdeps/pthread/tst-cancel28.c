/* Check if the thread created by POSIX timer using SIGEV_THREAD is
   cancellable.
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

#include <stdio.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>

#include <support/check.h>
#include <support/test-driver.h>
#include <support/xthread.h>

static pthread_barrier_t barrier;
static pthread_t timer_thread;

static void
cl (void *arg)
{
  xpthread_barrier_wait (&barrier);
}

static void
thread_handler (union sigval sv)
{
  timer_thread = pthread_self ();

  xpthread_barrier_wait (&barrier);

  pthread_cleanup_push (cl, NULL);
  while (1)
    clock_nanosleep (CLOCK_REALTIME, 0, &(struct timespec) { 1, 0 }, NULL);
  pthread_cleanup_pop (0);
}

static int
do_test (void)
{
  struct sigevent sev = { 0 };
  sev.sigev_notify = SIGEV_THREAD;
  sev.sigev_notify_function = &thread_handler;

  timer_t timerid;
  TEST_COMPARE (timer_create (CLOCK_REALTIME, &sev, &timerid), 0);

  xpthread_barrier_init (&barrier, NULL, 2);

  struct itimerspec trigger = { 0 };
  trigger.it_value.tv_nsec = 1000000;
  TEST_COMPARE (timer_settime (timerid, 0, &trigger, NULL), 0);

  xpthread_barrier_wait (&barrier);

  xpthread_cancel (timer_thread);

  xpthread_barrier_init (&barrier, NULL, 2);
  xpthread_barrier_wait (&barrier);

  return 0;
}

/* A stall in cancellation is a regression.  */
#include <support/test-driver.c>
