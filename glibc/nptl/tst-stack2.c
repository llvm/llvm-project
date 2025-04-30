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

/* Test whether it is possible to create a thread with PTHREAD_STACK_MIN
   stack size.  */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

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
  pthread_attr_t attr;
  pthread_attr_init (&attr);

  int result = 0;
  int res = pthread_attr_setstacksize (&attr, PTHREAD_STACK_MIN);
  if (res)
    {
      printf ("pthread_attr_setstacksize failed %d\n", res);
      result = 1;
    }

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

  if (seen != 1)
    {
      printf ("seen %d != 1\n", seen);
      result = 1;
    }

  return result;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
