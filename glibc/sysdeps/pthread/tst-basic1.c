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
#include <unistd.h>
#include <sys/types.h>


static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static pid_t pid;

static void *
tf (void *a)
{
  if (getpid () != pid)
    {
      write_message ("pid mismatch\n");
      _exit (1);
    }

  return a;
}


int
do_test (void)
{
  pid = getpid ();

#define N 2
  pthread_t t[N];
  int i;

  for (i = 0; i < N; ++i)
    if (pthread_create (&t[i], NULL, tf, (void *) (long int) (i + 1)) != 0)
      {
	write_message ("create failed\n");
	_exit (1);
      }
    else
      printf ("created thread %d\n", i);

  for (i = 0; i < N; ++i)
    {
      void *r;
      int e;
      if ((e = pthread_join (t[i], &r)) != 0)
	{
	  printf ("join failed: %d\n", e);
	  _exit (1);
	}
      else if (r != (void *) (long int) (i + 1))
	{
	  write_message ("result wrong\n");
	  _exit (1);
	}
      else
	printf ("joined thread %d\n", i);
    }

  return 0;
}
