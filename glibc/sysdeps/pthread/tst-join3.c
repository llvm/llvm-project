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
#include <string.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>
#include <support/xtime.h>


#define CLOCK_USE_TIMEDJOIN (-1)

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


static void *
tf (void *arg)
{
  xpthread_mutex_lock (&lock);
  xpthread_mutex_unlock (&lock);

  return (void *) 42l;
}


static int
do_test_clock (clockid_t clockid)
{
  const clockid_t clockid_for_get =
    (clockid == CLOCK_USE_TIMEDJOIN) ? CLOCK_REALTIME : clockid;

  xpthread_mutex_lock (&lock);
  pthread_t th = xpthread_create (NULL, tf, NULL);

  void *status;
  struct timespec timeout = timespec_add (xclock_now (clockid_for_get),
                                          make_timespec (0, 200000000));

  int val;
  if (clockid == CLOCK_USE_TIMEDJOIN)
    val = pthread_timedjoin_np (th, &status, &timeout);
  else
    val = pthread_clockjoin_np (th, &status, clockid, &timeout);

  TEST_COMPARE (val, ETIMEDOUT);

  xpthread_mutex_unlock (&lock);

  while (1)
    {
      timeout = timespec_add (xclock_now (clockid_for_get),
                              make_timespec (0, 200000000));

      if (clockid == CLOCK_USE_TIMEDJOIN)
        val = pthread_timedjoin_np (th, &status, &timeout);
      else
        val = pthread_clockjoin_np (th, &status, clockid, &timeout);
      if (val == 0)
	break;

      TEST_COMPARE (val, ETIMEDOUT);
    }

  if (status != (void *) 42l)
    FAIL_EXIT1 ("return value %p, expected %p\n", status, (void *) 42l);

  return 0;
}

static int
do_test (void)
{
  do_test_clock (CLOCK_USE_TIMEDJOIN);
  do_test_clock (CLOCK_REALTIME);
  do_test_clock (CLOCK_MONOTONIC);
  return 0;
}

#include <support/test-driver.c>
