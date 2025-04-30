/* Test for sem_post race: bug 14532.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <semaphore.h>
#include <stdio.h>

#define NTHREADS 10
#define NITER 100000

sem_t sem;
int c;
volatile int thread_fail;

static void *
tf (void *arg)
{
  for (int i = 0; i < NITER; i++)
    {
      if (sem_wait (&sem) != 0)
	{
	  perror ("sem_wait");
	  thread_fail = 1;
	}
      ++c;
      if (sem_post (&sem) != 0)
	{
	  perror ("sem_post");
	  thread_fail = 1;
	}
    }
  return NULL;
}

static int
do_test (void)
{
  if (sem_init (&sem, 0, 0) != 0)
    {
      perror ("sem_init");
      return 1;
    }

  pthread_t th[NTHREADS];
  for (int i = 0; i < NTHREADS; i++)
    {
      if (pthread_create (&th[i], NULL, tf, NULL) != 0)
	{
	  puts ("pthread_create failed");
	  return 1;
	}
    }

  if (sem_post (&sem) != 0)
    {
      perror ("sem_post");
      return 1;
    }

  for (int i = 0; i < NTHREADS; i++)
    if (pthread_join (th[i], NULL) != 0)
      {
	puts ("pthread_join failed");
	return 1;
      }

  if (c != NTHREADS * NITER)
    {
      printf ("c = %d, should be %d\n", c, NTHREADS * NITER);
      return 1;
    }
  return thread_fail;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
