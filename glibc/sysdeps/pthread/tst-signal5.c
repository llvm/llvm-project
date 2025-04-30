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

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>


static sigset_t ss;


static void *
tf (void *arg)
{
  sigset_t ss2;
  if (pthread_sigmask (SIG_SETMASK, NULL, &ss2) != 0)
    {
      puts ("child: sigmask failed");
      exit (1);
    }

  int i;
  for (i = 1; i < 32; ++i)
    if (sigismember (&ss, i) && ! sigismember (&ss2, i))
      {
	printf ("signal %d set in parent mask, but not in child\n", i);
	exit (1);
      }
    else if (! sigismember (&ss, i) && sigismember (&ss2, i))
      {
	printf ("signal %d set in child mask, but not in parent\n", i);
	exit (1);
      }

  return NULL;
}


static int
do_test (void)
{
  sigemptyset (&ss);
  sigaddset (&ss, SIGUSR1);
  if (pthread_sigmask (SIG_SETMASK, &ss, NULL) != 0)
    {
      puts ("1st sigmask failed");
      exit (1);
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("1st create failed");
      exit (1);
    }

  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("1st join failed");
      exit (1);
    }

  sigemptyset (&ss);
  sigaddset (&ss, SIGUSR2);
  sigaddset (&ss, SIGFPE);
  if (pthread_sigmask (SIG_SETMASK, &ss, NULL) != 0)
    {
      puts ("2nd sigmask failed");
      exit (1);
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("2nd create failed");
      exit (1);
    }

  if (pthread_join (th, &r) != 0)
    {
      puts ("2nd join failed");
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
