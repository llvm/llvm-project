/* Tests barrier initialization.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>


static int
do_test (void)
{
  pthread_barrier_t b;
  int e;
  int cnt;

  e = pthread_barrier_init (&b, NULL, 0);
  if (e == 0)
    {
      puts ("barrier_init with count 0 succeeded");
      return 1;
    }
  if (e != EINVAL)
    {
      puts ("barrier_init with count 0 didn't return EINVAL");
      return 1;
    }

  if (pthread_barrier_init (&b, NULL, 1) != 0)
    {
      puts ("real barrier_init failed");
      return 1;
    }

  for (cnt = 0; cnt < 10; ++cnt)
    {
      e = pthread_barrier_wait (&b);

      if (e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait didn't return PTHREAD_BARRIER_SERIAL_THREAD");
	  return 1;
	}
    }

  if (pthread_barrier_destroy (&b) != 0)
    {
      puts ("barrier_destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
