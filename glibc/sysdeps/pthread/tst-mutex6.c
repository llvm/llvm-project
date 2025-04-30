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
#include <unistd.h>
#include <errno.h>
#include <stdbool.h>

#ifndef TEST_FUNCTION
static int do_test (void);
# define TEST_FUNCTION do_test ()
#endif
#include "../test-skeleton.c"

#ifndef ATTR
pthread_mutexattr_t *attr;
# define ATTR attr
#endif

#ifndef ATTR_NULL
# define ATTR_NULL (ATTR == NULL)
#endif

static int
do_test (void)
{
  pthread_mutex_t m;

  int e = pthread_mutex_init (&m, ATTR);
  if (!ATTR_NULL && e == ENOTSUP)
    {
      puts ("cannot support selected type of mutexes");
      e = pthread_mutex_init (&m, NULL);
    }
  if (e != 0)
    {
      puts ("mutex_init failed");
      return 1;
    }

  if (!ATTR_NULL && pthread_mutexattr_destroy (ATTR) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("1st mutex_lock failed");
      return 1;
    }

  delayed_exit (1);
  /* This call should never return.  */
  xpthread_mutex_lock (&m);

  puts ("2nd mutex_lock returned");
  return 1;
}
