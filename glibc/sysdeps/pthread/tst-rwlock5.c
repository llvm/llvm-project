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

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_rwlock_t r;


static void *
tf (void *arg)
{
  if (pthread_rwlock_wrlock (&r) == 0)
    {
      puts ("child: rwlock_wrlock succeeded");
      exit (1);
    }

  puts ("child: rwlock_wrlock returned");

  exit (1);
}


static int
do_test (void)
{
  pthread_t th;

  if (pthread_rwlock_init (&r, NULL) != 0)
    {
      puts ("rwlock_init failed");
      return 1;
    }

  if (pthread_rwlock_wrlock (&r) != 0)
    {
      puts ("rwlock_wrlock failed");
      return 1;
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  delayed_exit (1);
  /* This call should never return.  */
  xpthread_mutex_lock (&m);

  puts ("2nd mutex_lock returned");
  return 1;
}
