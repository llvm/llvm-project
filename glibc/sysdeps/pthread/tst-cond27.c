/* Test pthread_cond_clockwait timeout.

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>


static pthread_mutex_t mut = PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;


static int
do_test_clock (clockid_t clockid)
{
  /* Get the mutex.  */
  xpthread_mutex_lock (&mut);

  /* Waiting for the condition will fail.  But we want the timeout here.  */
  const struct timespec ts_now = xclock_now (clockid);
  const struct timespec ts_timeout =
    timespec_add (ts_now, make_timespec (0, 500000000));

  /* In theory pthread_cond_clockwait could return zero here due to
     spurious wakeup. However that can't happen without a signal or an
     additional waiter.  */
  TEST_COMPARE (pthread_cond_clockwait (&cond, &mut, clockid, &ts_timeout),
                ETIMEDOUT);

  xpthread_mutex_unlock (&mut);

  return 0;
}

static int
do_test (void)
{
  do_test_clock (CLOCK_MONOTONIC);
  do_test_clock (CLOCK_REALTIME);
  return 0;
}

#include <support/test-driver.c>
