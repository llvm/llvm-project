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
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>


#define N 20

static pthread_t th[N];
static pthread_mutex_t lock[N];


static void *tf (void *a)
{
  uintptr_t idx = (uintptr_t) a;

  pthread_mutex_lock (&lock[idx]);

  return pthread_equal (pthread_self (), th[idx]) ? NULL : (void *) 1l;
}


int
do_test (void)
{
  if (pthread_equal (pthread_self (), pthread_self ()) == 0)
    {
      puts ("pthread_equal (pthread_self (), pthread_self ()) failed");
      exit (1);
    }

  pthread_attr_t at;

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

  int i;
  for (i = 0; i < N; ++i)
    {
      if (pthread_mutex_init (&lock[i], NULL) != 0)
	{
	  puts ("mutex_init failed");
	  exit (1);
	}

      if (pthread_mutex_lock (&lock[i]) != 0)
	{
	  puts ("mutex_lock failed");
	  exit (1);
	}

      if (pthread_create (&th[i], &at, tf, (void *) (long int) i) != 0)
	{
	  puts ("create failed");
	  exit (1);
	}

      if (pthread_mutex_unlock (&lock[i]) != 0)
	{
	  puts ("mutex_unlock failed");
	  exit (1);
	}

      printf ("created thread %d\n", i);
    }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  int result = 0;
  for (i = 0; i < N; ++i)
    {
      void *r;
      int e;
      if ((e = pthread_join (th[i], &r)) != 0)
	{
	  printf ("join failed: %d\n", e);
	  _exit (1);
	}
      else if (r != NULL)
	result = 1;
    }

  return result;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
