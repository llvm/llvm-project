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
#include <semaphore.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xtime.h>

/* A bogus clock value that tells run_test to use sem_timedwait rather than
   sem_clockwait.  */
#define CLOCK_USE_TIMEDWAIT (-1)

static void
do_test_clock (clockid_t clockid)
{
  const clockid_t clockid_for_get =
    clockid == CLOCK_USE_TIMEDWAIT ? CLOCK_REALTIME : clockid;
  sem_t s;
  struct timespec ts;

  TEST_COMPARE (sem_init (&s, 0, 1), 0);
  TEST_COMPARE (TEMP_FAILURE_RETRY (sem_wait (&s)), 0);

  /* We wait for half a second.  */
  xclock_gettime (clockid_for_get, &ts);
  ts = timespec_add (ts, make_timespec (0, TIMESPEC_HZ/2));

  errno = 0;
  TEST_COMPARE (TEMP_FAILURE_RETRY ((clockid == CLOCK_USE_TIMEDWAIT)
                                    ? sem_timedwait (&s, &ts)
                                    : sem_clockwait (&s, clockid, &ts)), -1);
  TEST_COMPARE (errno, ETIMEDOUT);
  TEST_TIMESPEC_NOW_OR_AFTER (clockid_for_get, ts);
}

static int do_test (void)
{
  do_test_clock (CLOCK_USE_TIMEDWAIT);
  do_test_clock (CLOCK_REALTIME);
  do_test_clock (CLOCK_MONOTONIC);
  return 0;
}

#include <support/test-driver.c>
