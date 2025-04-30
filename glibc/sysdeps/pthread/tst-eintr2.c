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
#include <support/xtime.h>

#include "eintr.c"


static pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t m2 = PTHREAD_MUTEX_INITIALIZER;


static void *
tf1 (void *arg)
{
  struct timespec ts = timespec_add (xclock_now (CLOCK_REALTIME),
                                     make_timespec (10000, 0));

  /* This call must never return.  */
  int e = pthread_mutex_timedlock (&m1, &ts);
  char buf[100];
  printf ("tf1: mutex_timedlock returned: %s\n",
	  strerror_r (e, buf, sizeof (buf)));

  exit (1);
}


static void *
tf2 (void *arg)
{
  while (1)
    {
      TEST_COMPARE (pthread_mutex_lock (&m2), 0);
      TEST_COMPARE (pthread_mutex_unlock (&m2), 0);

      struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000 };
      nanosleep (&ts, NULL);
    }
  return NULL;
}


static int
do_test (void)
{
  TEST_COMPARE (pthread_mutex_lock (&m1), 0);

  setup_eintr (SIGUSR1, NULL);

  char buf[100];
  xpthread_create (NULL, tf1, NULL);
  xpthread_create (NULL, tf2, NULL);

  delayed_exit (3);
  /* This call must never return.  */
  int e = pthread_mutex_lock (&m1);
  printf ("main: mutex_lock returned: %s\n",
	  strerror_r (e, buf, sizeof (buf)));

  return 1;
}

#include <support/test-driver.c>
