/* Test unsupported/bad clocks passed to pthread_mutex_clocklock.

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
#include <time.h>
#include <unistd.h>
#include <support/check.h>

static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static void test_bad_clockid (clockid_t clockid)
{
  const struct timespec ts = {0,0};
  TEST_COMPARE (pthread_mutex_clocklock (&mut, clockid, &ts), EINVAL);
}

#define NOT_A_VALID_CLOCK 123456

static int
do_test (void)
{
  /* These clocks are meaningless to pthread_mutex_clocklock.  */
#if defined(CLOCK_PROCESS_CPUTIME_ID)
  test_bad_clockid (CLOCK_PROCESS_CPUTIME_ID);
#endif
#if defined(CLOCK_THREAD_CPUTIME_ID)
  test_bad_clockid (CLOCK_PROCESS_CPUTIME_ID);
#endif

  /* These clocks might be meaningful, but are currently unsupported by
     pthread_mutex_clocklock.  */
#if defined(CLOCK_REALTIME_COARSE)
  test_bad_clockid (CLOCK_REALTIME_COARSE);
#endif
#if defined(CLOCK_MONOTONIC_RAW)
  test_bad_clockid (CLOCK_MONOTONIC_RAW);
#endif
#if defined(CLOCK_MONOTONIC_COARSE)
  test_bad_clockid (CLOCK_MONOTONIC_COARSE);
#endif
#if defined(CLOCK_BOOTTIME)
  test_bad_clockid (CLOCK_BOOTTIME);
#endif

  /* This is a completely invalid clock.  */
  test_bad_clockid (NOT_A_VALID_CLOCK);

  return 0;
}

#include <support/test-driver.c>
