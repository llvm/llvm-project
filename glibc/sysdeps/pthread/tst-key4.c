/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#ifdef PTHREAD_KEYS_MAX
const int max = PTHREAD_KEYS_MAX;
#else
const int max = _POSIX_THREAD_KEYS_MAX;
#endif
static pthread_key_t *keys;


static void *
tf1 (void *arg)
{
  int i;
  for (i = 0; i < max; ++i)
    if (pthread_setspecific (keys[i], (void *) (long int) (i + 1)) != 0)
      {
	puts ("setspecific failed");
	exit (1);
      }

  return NULL;
}


static void *
tf2 (void *arg)
{
  int i;
  for (i = 0; i < max; ++i)
    if (pthread_getspecific (keys[i]) != NULL)
      {
	printf ("getspecific for key %d not NULL\n", i);
	exit (1);
      }

  return NULL;
}


static int
do_test (void)
{
  keys = alloca (max * sizeof (pthread_key_t));

  int i;
  for (i = 0; i < max; ++i)
    if (pthread_key_create (&keys[i], NULL) != 0)
      {
	puts ("key_create failed");
	exit (1);
      }

  pthread_attr_t a;

  if (pthread_attr_init (&a) != 0)
    {
      puts ("attr_init failed");
      exit (1);
    }

  if (pthread_attr_setstacksize (&a, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

  for (i = 0; i < 10; ++i)
    {
      int j;
#define N 2
      pthread_t th[N];
      for (j = 0; j < N; ++j)
	if (pthread_create (&th[j], NULL, tf1, NULL) != 0)
	  {
	    puts ("1st create failed");
	    exit (1);
	  }

      for (j = 0; j < N; ++j)
	if (pthread_join (th[j], NULL) != 0)
	  {
	    puts ("1st join failed");
	    exit (1);
	  }

      for (j = 0; j < N; ++j)
	if (pthread_create (&th[j], NULL, tf2, NULL) != 0)
	  {
	    puts ("2nd create failed");
	    exit (1);
	  }

      for (j = 0; j < N; ++j)
	if (pthread_join (th[j], NULL) != 0)
	  {
	    puts ("2nd join failed");
	    exit (1);
	  }
    }

  if (pthread_attr_destroy (&a) != 0)
    {
      puts ("attr_destroy failed");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
