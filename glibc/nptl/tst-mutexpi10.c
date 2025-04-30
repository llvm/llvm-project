/* Check if pthread_mutex_clocklock with PRIO_INHERIT fails with clock
   different than CLOCK_REALTIME.
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
#include <errno.h>
#include <array_length.h>

#include <support/check.h>
#include <support/xthread.h>
#include <support/timespec.h>

static int
do_test (void)
{
  const int types[] = {
    PTHREAD_MUTEX_NORMAL,
    PTHREAD_MUTEX_ERRORCHECK,
    PTHREAD_MUTEX_RECURSIVE,
    PTHREAD_MUTEX_ADAPTIVE_NP
  };
  const int robust[] = {
    PTHREAD_MUTEX_STALLED,
    PTHREAD_MUTEX_ROBUST
  };


  for (int t = 0; t < array_length (types); t++)
    for (int r = 0; r < array_length (robust); r++)
      {
	pthread_mutexattr_t attr;

	xpthread_mutexattr_init (&attr);
	xpthread_mutexattr_setprotocol (&attr, PTHREAD_PRIO_INHERIT);
	xpthread_mutexattr_settype (&attr, types[t]);
	xpthread_mutexattr_setrobust (&attr, robust[r]);

	pthread_mutex_t mtx;
	xpthread_mutex_init (&mtx, &attr);

	struct timespec tmo = timespec_add (xclock_now (CLOCK_MONOTONIC),
					    make_timespec (0, 100000000));

	TEST_COMPARE (pthread_mutex_clocklock (&mtx, CLOCK_MONOTONIC, &tmo),
		      EINVAL);

	xpthread_mutex_destroy (&mtx);
      }

  return 0;
}

#include <support/test-driver.c>
