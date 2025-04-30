/* Test that pthread_mutex_timedlock properly times out.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

pthread_mutex_t mutex;

static void *
thr (void *arg)
{
  struct timespec abstime;
  clock_gettime (CLOCK_REALTIME, &abstime);
  abstime.tv_sec += 1;
  int ret = pthread_mutex_timedlock (&mutex, &abstime);
  if (ret == 0)
    {
      puts ("mutex_timedlock didn't fail");
      exit (1);
    }
  if (ret != ETIMEDOUT)
    {
      printf ("mutex_timedlock failed: %s\n", strerror (ret));
      exit (1);
    }

  return 0;
}

static int
do_test (void)
{
  pthread_t pt;
  pthread_mutexattr_t ma;

  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("mutexattr_init failed");
      return 0;
    }
  if (pthread_mutexattr_setrobust (&ma, PTHREAD_MUTEX_ROBUST_NP) != 0)
    {
      puts ("mutexattr_setrobust failed");
      return 1;
    }
  if (pthread_mutex_init (&mutex, &ma))
    {
      puts ("mutex_init failed");
      return 1;
    }

  if (pthread_mutexattr_destroy (&ma))
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  if (pthread_mutex_lock (&mutex))
    {
      puts ("mutex_lock failed");
      return 1;
    }

  if (pthread_create (&pt, NULL, thr, NULL))
    {
      puts ("pthread_create failed");
      return 1;
    }

  if (pthread_join (pt, NULL))
    {
      puts ("pthread_join failed");
      return 1;
    }

  if (pthread_mutex_unlock (&mutex))
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  if (pthread_mutex_destroy (&mutex))
    {
      puts ("mutex_destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
