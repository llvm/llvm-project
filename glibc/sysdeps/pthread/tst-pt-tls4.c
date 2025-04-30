/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 3

void (*test1) (void), (*test2) (void);

pthread_barrier_t b2, b3;

static void *
tf (void *arg)
{
  int i;

  for (i = 0; i <= (uintptr_t) arg; ++i)
    {
      int r = pthread_barrier_wait (&b3);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("tf: barrier_wait failed");
	  exit (1);
	}
    }

  test1 ();

  for (i = 0; i < 3; ++i)
    {
      int r = pthread_barrier_wait (&b3);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("tf: barrier_wait failed");
	  exit (1);
	}
    }

  test2 ();

  for (i = 0; i < 3 - (uintptr_t) arg; ++i)
    {
      int r = pthread_barrier_wait (&b3);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("tf: barrier_wait failed");
	  exit (1);
	}
    }

  return NULL;
}

static void *
tf2 (void *arg)
{
  int r = pthread_barrier_wait (&b2);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("tf2: barrier_wait failed");
      exit (1);
    }

  int i;
  for (i = 0; i < N; ++i)
    tf (arg);
  return NULL;
}

int
do_test (void)
{
  pthread_t th[2];
  const char *modules[N]
    = { "tst-tls4moda.so", "tst-tls4moda.so", "tst-tls4modb.so" };

  if (pthread_barrier_init (&b2, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_barrier_init (&b3, NULL, 3) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_create (&th[0], NULL, tf2, (void *) (uintptr_t) 1))
    {
      puts ("pthread_create failed");
      return 1;
    }

  int r = pthread_barrier_wait (&b2);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  int i;
  for (i = 0; i < N; ++i)
    {
      void *h = dlopen (modules[i], RTLD_LAZY);
      if (h == NULL)
	{
	  printf ("dlopen failed %s\n", dlerror ());
	  return 1;
	}

      test1 = dlsym (h, "test1");
      if (test1 == NULL)
	{
	  printf ("dlsym for test1 failed %s\n", dlerror ());
	  return 1;
	}

      test2 = dlsym (h, "test2");
      if (test2 == NULL)
	{
	  printf ("dlsym for test2 failed %s\n", dlerror ());
	  return 1;
	}

      if (pthread_create (&th[1], NULL, tf, (void *) (uintptr_t) 2))
	{
	  puts ("pthread_create failed");
	  return 1;
	}

      tf ((void *) (uintptr_t) 0);

      if (pthread_join (th[1], NULL) != 0)
	{
	  puts ("join failed");
	  return 1;
	}

      if (dlclose (h))
	{
	  puts ("dlclose failed");
	  return 1;
	}

      printf ("test %d with %s succeeded\n", i, modules[i]);
    }

  if (pthread_join (th[0], NULL) != 0)
    {
      puts ("join failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
