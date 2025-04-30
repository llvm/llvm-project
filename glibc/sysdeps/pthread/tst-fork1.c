/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Roland McGrath <roland@redhat.com>, 2002.

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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

static void *
thread_function (void * arg)
{
  int i = (intptr_t) arg;
  int status;
  pid_t pid;
  pid_t pid2;

  pid = fork ();
  switch (pid)
    {
    case 0:
      printf ("%ld for %d\n", (long int) getpid (), i);
      struct timespec ts = { .tv_sec = 0, .tv_nsec = 100000000 * i };
      nanosleep (&ts, NULL);
      _exit (i);
      break;
    case -1:
      printf ("fork: %m\n");
      return (void *) 1l;
      break;
    }

  pid2 = TEMP_FAILURE_RETRY (waitpid (pid, &status, 0));
  if (pid2 != pid)
    {
      printf ("waitpid returned %ld, expected %ld\n",
	      (long int) pid2, (long int) pid);
      return (void *) 1l;
    }

  printf ("%ld with %d, expected %d\n",
	  (long int) pid, WEXITSTATUS (status), i);

  return WEXITSTATUS (status) == i ? NULL : (void *) 1l;
}

#define N 5
static const int t[N] = { 7, 6, 5, 4, 3 };

static int
do_test (void)
{
  pthread_t th[N];
  int i;
  int result = 0;
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

  for (i = 0; i < N; ++i)
    if (pthread_create (&th[i], NULL, thread_function,
			(void *) (intptr_t) t[i]) != 0)
      {
	printf ("creation of thread %d failed\n", i);
	exit (1);
      }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  for (i = 0; i < N; ++i)
    {
      void *v;
      if (pthread_join (th[i], &v) != 0)
	{
	  printf ("join of thread %d failed\n", i);
	  result = 1;
	}
      else if (v != NULL)
	{
	  printf ("join %d successful, but child failed\n", i);
	  result = 1;
	}
      else
	printf ("join %d successful\n", i);
    }

  return result;
}

#include <support/test-driver.c>
