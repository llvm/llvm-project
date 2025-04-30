/* Test that pthread_create diagnoses invalid scheduling parameters.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void *
thread_function (void *arg)
{
  abort ();
}


static int
do_test (void)
{
#if !defined SCHED_FIFO || !defined SCHED_OTHER
  puts ("SCHED_FIFO or SCHED_OTHER not available at compile time");
  return 0; /* 77 */
#else

  int err;

#define TRY(func, arglist)                              \
  if ((err = func arglist) != 0)                        \
    {                                                   \
      printf ("%s: %s\n", #func, strerror (err));       \
      return 2;                                         \
    }

  int fifo_max = sched_get_priority_max (SCHED_FIFO);
  if (fifo_max == -1)
    {
      assert (errno == ENOTSUP || errno == ENOSYS);
      puts ("SCHED_FIFO not supported, cannot test");
      return 0; /* 77 */
    }

  int other_max = sched_get_priority_max (SCHED_OTHER);
  if (other_max == -1)
    {
      assert (errno == ENOTSUP || errno == ENOSYS);
      puts ("SCHED_OTHER not supported, cannot test");
      return 0; /* 77 */
    }

  assert (fifo_max > other_max);

  pthread_attr_t attr;
  TRY (pthread_attr_init, (&attr));
  TRY (pthread_attr_setinheritsched, (&attr, PTHREAD_EXPLICIT_SCHED));
  TRY (pthread_attr_setschedpolicy, (&attr, SCHED_FIFO));

  /* This value is chosen so as to be valid for SCHED_FIFO but invalid for
     SCHED_OTHER.  */
  struct sched_param param = { .sched_priority = other_max + 1 };
  TRY (pthread_attr_setschedparam, (&attr, &param));

  TRY (pthread_attr_setschedpolicy, (&attr, SCHED_OTHER));

  /* Now ATTR has a sched_param that is invalid for its policy.  */
  pthread_t th;
  err = pthread_create (&th, &attr, &thread_function, NULL);
  if (err != EINVAL)
    {
      printf ("pthread_create returned %d (%s), expected %d (EINVAL: %s)\n",
              err, strerror (err), EINVAL, strerror (EINVAL));
      return 1;
    }

  return 0;
#endif
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
