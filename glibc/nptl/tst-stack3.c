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

/* Test whether pthread_create/pthread_join with user defined stacks
   doesn't leak memory.
   NOTE: this tests functionality beyond POSIX.  In POSIX user defined
   stacks cannot be ever freed once used by pthread_create nor they can
   be reused for other thread.  */

#include <limits.h>
#include <mcheck.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int seen;

static void *
tf (void *p)
{
  ++seen;
  return NULL;
}

static int
do_test (void)
{
  mtrace ();

  void *stack;
  int res = posix_memalign (&stack, getpagesize (), 4 * PTHREAD_STACK_MIN);
  if (res)
    {
      printf ("malloc failed %s\n", strerror (res));
      return 1;
    }

  pthread_attr_t attr;
  pthread_attr_init (&attr);

  int result = 0;
  res = pthread_attr_setstack (&attr, stack, 4 * PTHREAD_STACK_MIN);
  if (res)
    {
      printf ("pthread_attr_setstack failed %d\n", res);
      result = 1;
    }

  for (int i = 0; i < 16; ++i)
    {
      /* Create the thread.  */
      pthread_t th;
      res = pthread_create (&th, &attr, tf, NULL);
      if (res)
	{
	  printf ("pthread_create failed %d\n", res);
	  result = 1;
	}
      else
	{
	  res = pthread_join (th, NULL);
	  if (res)
	    {
	      printf ("pthread_join failed %d\n", res);
	      result = 1;
	    }
	}
    }

  pthread_attr_destroy (&attr);

  if (seen != 16)
    {
      printf ("seen %d != 16\n", seen);
      result = 1;
    }

  free (stack);
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
