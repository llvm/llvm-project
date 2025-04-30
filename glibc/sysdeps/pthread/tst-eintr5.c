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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>
#include <support/xtime.h>

#include "eintr.c"


static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t c = PTHREAD_COND_INITIALIZER;


static void *
tf (void *arg)
{
  struct timespec ts = timespec_add (xclock_now (CLOCK_REALTIME),
                                     make_timespec (10000, 0));

  /* This call must never return.  */
  TEST_COMPARE (pthread_cond_timedwait (&c, &m, &ts), 0);
  FAIL_EXIT1 ("pthread_cond_timedwait returned unexpectedly\n");
}


static int
do_test (void)
{
  setup_eintr (SIGUSR1, NULL);

  xpthread_create (NULL, tf, NULL);

  delayed_exit (3);
  /* This call must never return.  */
  xpthread_cond_wait (&c, &m);
  FAIL_RET ("error: pthread_cond_wait returned");
}

#include <support/test-driver.c>
