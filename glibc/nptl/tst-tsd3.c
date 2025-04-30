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


static pthread_key_t key1;
static pthread_key_t key2;


static int left;


static void
destr1 (void *arg)
{
  if (--left > 0)
    {
      puts ("set key2");

      /* Use an arbirary but valid pointer to avoid GCC warnings.  */
      if (pthread_setspecific (key2, (void *) &left) != 0)
	{
	  puts ("destr1: setspecific failed");
	  exit (1);
	}
    }
}


static void
destr2 (void *arg)
{
  if (--left > 0)
    {
      puts ("set key1");

      /* Use an arbirary but valid pointer to avoid GCC warnings.  */
      if (pthread_setspecific (key1, (void *) &left) != 0)
	{
	  puts ("destr2: setspecific failed");
	  exit (1);
	}
    }
}


static void *
tf (void *arg)
{
  /* Let the destructors work.  */
  left = 7;

  /* Use an arbirary but valid pointer to avoid GCC warnings.  */
  if (pthread_setspecific (key1, (void *) &left) != 0
      || pthread_setspecific (key2, (void *) &left) != 0)
    {
      puts ("tf: setspecific failed");
      exit (1);
    }

  return NULL;
}


static int
do_test (void)
{
  /* Allocate two keys, both with destructors.  */
  if (pthread_key_create (&key1, destr1) != 0
      || pthread_key_create (&key2, destr2) != 0)
    {
      puts ("key_create failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  if (pthread_join (th, NULL) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (left != 0)
    {
      printf ("left == %d\n", left);
      return 1;
    }

  if (pthread_getspecific (key1) != NULL)
    {
      puts ("key1 data != NULL");
      return 1;
    }
  if (pthread_getspecific (key2) != NULL)
    {
      puts ("key2 data != NULL");
      return 1;
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
