/* Test unsupported/bad clocks passed to sem_clockwait.

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
#include <semaphore.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>


#define NOT_A_VALID_CLOCK 123456

static int
do_test (void)
{
  sem_t s;
  TEST_COMPARE (sem_init (&s, 0, 1), 0);

  const struct timespec ts = make_timespec (0, 0);

  /* These clocks are meaningless to sem_clockwait.  */
#if defined(CLOCK_PROCESS_CPUTIME_ID)
  TEST_COMPARE (sem_clockwait (&s, CLOCK_PROCESS_CPUTIME_ID, &ts), -1);
  TEST_COMPARE (errno, EINVAL);
#endif
#if defined(CLOCK_THREAD_CPUTIME_ID)
  TEST_COMPARE (sem_clockwait (&s, CLOCK_THREAD_CPUTIME_ID, &ts), -1);
  TEST_COMPARE (errno, EINVAL);
#endif

  /* These clocks might be meaningful, but are currently unsupported
     by pthread_cond_clockwait.  */
#if defined(CLOCK_REALTIME_COARSE)
  TEST_COMPARE (sem_clockwait (&s, CLOCK_REALTIME_COARSE, &ts), -1);
  TEST_COMPARE (errno, EINVAL);
#endif
#if defined(CLOCK_MONOTONIC_RAW)
  TEST_COMPARE (sem_clockwait (&s, CLOCK_MONOTONIC_RAW, &ts), -1);
  TEST_COMPARE (errno, EINVAL);
#endif
#if defined(CLOCK_MONOTONIC_COARSE)
  TEST_COMPARE (sem_clockwait (&s, CLOCK_MONOTONIC_COARSE, &ts), -1);
  TEST_COMPARE (errno, EINVAL);
#endif
#if defined(CLOCK_BOOTTIME)
  TEST_COMPARE (sem_clockwait (&s, CLOCK_BOOTTIME, &ts), -1);
  TEST_COMPARE (errno, EINVAL);
#endif

  /* This is a completely invalid clock.  */
  TEST_COMPARE (sem_clockwait (&s, NOT_A_VALID_CLOCK, &ts), -1);
  TEST_COMPARE (errno, EINVAL);

  return 0;
}

#include <support/test-driver.c>
