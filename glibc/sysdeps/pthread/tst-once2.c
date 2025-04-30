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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define N 100

static pthread_once_t once = PTHREAD_ONCE_INIT;

static int global;

static void
once_handler (void)
{
  struct timespec ts;

  ++global;

  ts.tv_sec = 2;
  ts.tv_nsec = 0;
  nanosleep (&ts, NULL);
}


static void *
tf (void *arg)
{
  pthread_once (&once, once_handler);

  if (global != 1)
    {
      printf ("thread %ld: global == %d\n", (long int) arg, global);
      exit (1);
    }

  return NULL;
}


static int
do_test (void)
{
  pthread_attr_t at;
  pthread_t th[N];
  int cnt;

  if (pthread_attr_init (&at) != 0)
    {
      puts ("attr_init failed");
      return 1;
    }

  if (pthread_attr_setstacksize (&at, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

  for (cnt = 0; cnt < N; ++cnt)
    if (pthread_create (&th[cnt], &at, tf, (void *) (long int) cnt) != 0)
      {
	printf ("creation of thread %d failed\n", cnt);
	return 1;
      }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  for (cnt = 0; cnt < N; ++cnt)
    if (pthread_join (th[cnt], NULL) != 0)
      {
	printf ("join of thread %d failed\n", cnt);
	return 1;
      }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
