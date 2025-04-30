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

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int nrunning = 1;


static void
final_test (void)
{
  puts ("final_test has been called");

#define THE_SIGNAL SIGUSR1
  kill (getpid (), SIGUSR1);
}


static void *
tf (void *a)
{
  if (pthread_join ((pthread_t) a, NULL) != 0)
    {
      printf ("join failed while %d are running\n", nrunning);
      _exit (1);
    }

  printf ("%2d left\n", --nrunning);

  return NULL;
}


int
do_test (void)
{
#define N 20
  pthread_t t[N];
  pthread_t last = pthread_self ();
  int i;

  atexit (final_test);

  printf ("starting %d + 1 threads\n", N);
  for (i = 0; i < N; ++i)
    {
      if (pthread_create (&t[i], NULL, tf, (void *) last) != 0)
	{
	  puts ("create failed");
	  _exit (1);
	}

      ++nrunning;

      last = t[i];
    }

  printf ("%2d left\n", --nrunning);

  pthread_exit (NULL);
}


#define EXPECTED_SIGNAL THE_SIGNAL
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
