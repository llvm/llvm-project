/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tst-tpp.h"

static int
do_test (void)
{
  int ret = 0;

  init_tpp_test ();

  pthread_mutexattr_t ma;
  if (pthread_mutexattr_init (&ma))
    {
      puts ("mutexattr_init failed");
      return 1;
    }
  if (pthread_mutexattr_setprotocol (&ma, PTHREAD_PRIO_PROTECT))
    {
      puts ("mutexattr_setprotocol failed");
      return 1;
    }

  int prioceiling;
  if (pthread_mutexattr_getprioceiling (&ma, &prioceiling))
    {
      puts ("mutexattr_getprioceiling failed");
      return 1;
    }

  if (prioceiling < fifo_min || prioceiling > fifo_max)
    {
      printf ("prioceiling %d not in %d..%d range\n",
	      prioceiling, fifo_min, fifo_max);
      return 1;
    }

  if (fifo_max < INT_MAX
      && pthread_mutexattr_setprioceiling (&ma, fifo_max + 1) != EINVAL)
    {
      printf ("mutexattr_setprioceiling %d did not fail with EINVAL\n",
	      fifo_max + 1);
      return 1;
    }

  if (fifo_min > 0
      && pthread_mutexattr_setprioceiling (&ma, fifo_min - 1) != EINVAL)
    {
      printf ("mutexattr_setprioceiling %d did not fail with EINVAL\n",
	      fifo_min - 1);
      return 1;
    }

  if (pthread_mutexattr_setprioceiling (&ma, fifo_min))
    {
      puts ("mutexattr_setprioceiling failed");
      return 1;
    }

  if (pthread_mutexattr_setprioceiling (&ma, fifo_max))
    {
      puts ("mutexattr_setprioceiling failed");
      return 1;
    }

  if (pthread_mutexattr_setprioceiling (&ma, 6))
    {
      puts ("mutexattr_setprioceiling failed");
      return 1;
    }

  if (pthread_mutexattr_getprioceiling (&ma, &prioceiling))
    {
      puts ("mutexattr_getprioceiling failed");
      return 1;
    }

  if (prioceiling != 6)
    {
      printf ("mutexattr_getprioceiling returned %d != 6\n",
	      prioceiling);
      return 1;
    }

  pthread_mutex_t m1, m2, m3;
  int e = pthread_mutex_init (&m1, &ma);
  if (e == ENOTSUP)
    {
      puts ("cannot support selected type of mutexes");
      return 0;
    }
  else if (e != 0)
    {
      puts ("mutex_init failed");
      return 1;
    }

  if (pthread_mutexattr_setprioceiling (&ma, 8))
    {
      puts ("mutexattr_setprioceiling failed");
      return 1;
    }

  if (pthread_mutex_init (&m2, &ma))
    {
      puts ("mutex_init failed");
      return 1;
    }

  if (pthread_mutexattr_setprioceiling (&ma, 5))
    {
      puts ("mutexattr_setprioceiling failed");
      return 1;
    }

  if (pthread_mutex_init (&m3, &ma))
    {
      puts ("mutex_init failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 4);

  if (pthread_mutex_lock (&m1) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 6);

  if (pthread_mutex_trylock (&m2) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 8);

  if (pthread_mutex_lock (&m3) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 8);

  if (pthread_mutex_unlock (&m2) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 6);

  if (pthread_mutex_unlock (&m1) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 5);

  if (pthread_mutex_lock (&m2) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 8);

  if (pthread_mutex_unlock (&m2) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 5);

  if (pthread_mutex_getprioceiling (&m1, &prioceiling))
    {
      puts ("mutex_getprioceiling m1 failed");
      return 1;
    }
  else if (prioceiling != 6)
    {
      printf ("unexpected m1 prioceiling %d != 6\n", prioceiling);
      return 1;
    }

  if (pthread_mutex_getprioceiling (&m2, &prioceiling))
    {
      puts ("mutex_getprioceiling m2 failed");
      return 1;
    }
  else if (prioceiling != 8)
    {
      printf ("unexpected m2 prioceiling %d != 8\n", prioceiling);
      return 1;
    }

  if (pthread_mutex_getprioceiling (&m3, &prioceiling))
    {
      puts ("mutex_getprioceiling m3 failed");
      return 1;
    }
  else if (prioceiling != 5)
    {
      printf ("unexpected m3 prioceiling %d != 5\n", prioceiling);
      return 1;
    }

  if (pthread_mutex_setprioceiling (&m1, 7, &prioceiling))
    {
      printf ("mutex_setprioceiling failed");
      return 1;
    }
  else if (prioceiling != 6)
    {
      printf ("unexpected m1 old prioceiling %d != 6\n", prioceiling);
      return 1;
    }

  if (pthread_mutex_getprioceiling (&m1, &prioceiling))
    {
      puts ("mutex_getprioceiling m1 failed");
      return 1;
    }
  else if (prioceiling != 7)
    {
      printf ("unexpected m1 prioceiling %d != 7\n", prioceiling);
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 5);

  if (pthread_mutex_unlock (&m3) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 4);

  if (pthread_mutex_trylock (&m1) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (4, 7);

  struct sched_param sp;
  memset (&sp, 0, sizeof (sp));
  sp.sched_priority = 8;
  if (pthread_setschedparam (pthread_self (), SCHED_FIFO, &sp))
    {
      puts ("cannot set scheduling params");
      return 1;
    }

  CHECK_TPP_PRIORITY (8, 8);

  if (pthread_mutex_unlock (&m1) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  CHECK_TPP_PRIORITY (8, 8);

  if (pthread_mutex_lock (&m3) != EINVAL)
    {
      puts ("pthread_mutex_lock didn't fail with EINVAL");
      return 1;
    }

  CHECK_TPP_PRIORITY (8, 8);

  if (pthread_mutex_destroy (&m1) != 0)
    {
      puts ("mutex_destroy failed");
      return 1;
    }

  if (pthread_mutex_destroy (&m2) != 0)
    {
      puts ("mutex_destroy failed");
      return 1;
    }

  if (pthread_mutex_destroy (&m3) != 0)
    {
      puts ("mutex_destroy failed");
      return 1;
    }

  if (pthread_mutexattr_destroy (&ma) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
