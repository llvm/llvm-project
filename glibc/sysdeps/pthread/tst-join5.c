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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>
#include <support/xtime.h>

static void
wait_code (void)
{
  struct timespec ts = { .tv_sec = 0, .tv_nsec = 200000000 };
  while (nanosleep (&ts, &ts) < 0)
    ;
}


#ifdef WAIT_IN_CHILD
static pthread_barrier_t b;
#endif

static int
thread_join (pthread_t thread, void **retval)
{
#if defined USE_PTHREAD_TIMEDJOIN_NP
  const struct timespec ts = timespec_add (xclock_now (CLOCK_REALTIME),
                                           make_timespec (1000, 0));
  return pthread_timedjoin_np (thread, retval, &ts);
#elif defined USE_PTHREAD_CLOCKJOIN_NP_REALTIME
  const struct timespec ts = timespec_add (xclock_now (CLOCK_REALTIME),
                                           make_timespec (1000, 0));
  return pthread_clockjoin_np (thread, retval, CLOCK_REALTIME, &ts);
#elif defined USE_PTHREAD_CLOCKJOIN_NP_MONOTONIC
  const struct timespec ts = timespec_add (xclock_now (CLOCK_MONOTONIC),
                                           make_timespec (1000, 0));
  return pthread_clockjoin_np (thread, retval, CLOCK_MONOTONIC, &ts);
#else
  return pthread_join (thread, retval);
#endif
}


static void *
tf1 (void *arg)
{
#ifdef WAIT_IN_CHILD
  xpthread_barrier_wait (&b);

  wait_code ();
#endif

  thread_join ((pthread_t) arg, NULL);

  exit (42);
}


static void *
tf2 (void *arg)
{
#ifdef WAIT_IN_CHILD
  xpthread_barrier_wait (&b);

  wait_code ();
#endif

  thread_join ((pthread_t) arg, NULL);

  exit (43);
}


static int
do_test (void)
{
#ifdef WAIT_IN_CHILD
  xpthread_barrier_init (&b, NULL, 2);
#endif

  pthread_t th;

  int err = thread_join (pthread_self (), NULL);
  if (err == 0)
    {
      puts ("1st circular join succeeded");
      return 1;
    }
  if (err != EDEADLK)
    {
      printf ("1st circular join %d, not EDEADLK\n", err);
      return 1;
    }

  th = xpthread_create (NULL, tf1, (void *) pthread_self ());

#ifndef WAIT_IN_CHILD
  wait_code ();
#endif

  xpthread_cancel (th);

#ifdef WAIT_IN_CHILD
  xpthread_barrier_wait (&b);
#endif

  void *r;
  err = thread_join (th, &r);
  if (err != 0)
    {
      printf ("cannot join 1st thread: %d\n", err);
      return 1;
    }
  if (r != PTHREAD_CANCELED)
    {
      puts ("1st thread not canceled");
      return 1;
    }

  err = thread_join (pthread_self (), NULL);
  if (err == 0)
    {
      puts ("2nd circular join succeeded");
      return 1;
    }
  if (err != EDEADLK)
    {
      printf ("2nd circular join %d, not EDEADLK\n", err);
      return 1;
    }

  th = xpthread_create (NULL, tf2, (void *) pthread_self ());

#ifndef WAIT_IN_CHILD
  wait_code ();
#endif

  xpthread_cancel (th);

#ifdef WAIT_IN_CHILD
  xpthread_barrier_wait (&b);
#endif

  if (thread_join (th, &r) != 0)
    {
      puts ("cannot join 2nd thread");
      return 1;
    }
  if (r != PTHREAD_CANCELED)
    {
      puts ("2nd thread not canceled");
      return 1;
    }

  err = thread_join (pthread_self (), NULL);
  if (err == 0)
    {
      puts ("3rd circular join succeeded");
      return 1;
    }
  if (err != EDEADLK)
    {
      printf ("3rd circular join %d, not EDEADLK\n", err);
      return 1;
    }

  return 0;
}

#include <support/test-driver.c>
