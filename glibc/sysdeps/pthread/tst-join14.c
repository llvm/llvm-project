/* pthread_timedjoin_np, pthread_clockjoin_np NULL timeout test.
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
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>
#include <support/xtime.h>


#define CLOCK_USE_TIMEDJOIN (-1)


static void *
tf (void *arg)
{
  struct timespec ts = make_timespec(0, 100000);
  nanosleep(&ts, NULL);

  return (void *) 42l;
}


/* Check that pthread_timedjoin_np and pthread_clockjoin_np wait "forever" if
 * passed a timeout parameter of NULL. We can't actually wait forever, but we
 * can be sure that we did at least wait for some time by checking the exit
 * status of the thread. */
static int
do_test_clock (clockid_t clockid)
{
  pthread_t th = xpthread_create (NULL, tf, NULL);

  void *status;
  int val = (clockid == CLOCK_USE_TIMEDJOIN)
    ? pthread_timedjoin_np (th, &status, NULL)
    : pthread_clockjoin_np (th, &status, clockid, NULL);
  TEST_COMPARE (val, 0);

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
