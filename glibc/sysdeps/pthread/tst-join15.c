/* Check pthread_clockjoin_np clock support.
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

#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#include <array_length.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>

static void *
tf (void *arg)
{
  pause ();
  return NULL;
}


static int
do_test (void)
{
  const clockid_t clocks[] = {
    CLOCK_REALTIME,
    CLOCK_MONOTONIC,
    CLOCK_PROCESS_CPUTIME_ID,
    CLOCK_THREAD_CPUTIME_ID,
    CLOCK_THREAD_CPUTIME_ID,
    CLOCK_MONOTONIC_RAW,
    CLOCK_REALTIME_COARSE,
    CLOCK_MONOTONIC_COARSE,
#ifdef CLOCK_BOOTTIME
    CLOCK_BOOTTIME,
#endif
#ifdef CLOCK_REALTIME_ALARM
    CLOCK_REALTIME_ALARM,
#endif
#ifdef CLOCK_BOOTTIME_ALARM
    CLOCK_BOOTTIME_ALARM,
#endif
#ifdef CLOCK_TAI
    CLOCK_TAI
#endif
  };

  pthread_t thr = xpthread_create (NULL, tf, NULL);

  for (int t = 0; t < array_length (clocks); t++)
    {
      /* Create a valid timeout to check for ETIMEDOUT on valid clocks.  */
      struct timespec tmo;
      if (clock_gettime (clocks[t], &tmo) == -1)
	/* For clocks not supported, create a large timeout (it should
	   fail early with EINVAL).  */
	tmo = make_timespec (-1, 0);
      else
	tmo = timespec_add (tmo, make_timespec (0, 100000000));

      int ret = clocks[t] == CLOCK_REALTIME || clocks[t] == CLOCK_MONOTONIC
		? ETIMEDOUT : EINVAL;

      TEST_COMPARE (pthread_clockjoin_np (thr, NULL, clocks[t], &tmo), ret);
    }

  return 0;
}

#include <support/test-driver.c>
