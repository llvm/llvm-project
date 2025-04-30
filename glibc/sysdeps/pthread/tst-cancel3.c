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
#include <signal.h>
#include <stdio.h>
#include <unistd.h>


static int fd[2];


static void *
tf (void *arg)
{
  char buf[100];

  if (read (fd[0], buf, sizeof (buf)) == sizeof (buf))
    {
      puts ("read succeeded");
      return (void *) 1l;
    }

  return NULL;
}


static int
do_test (void)
{
  pthread_t th;
  void *r;
  struct sigaction sa;

  sa.sa_handler = SIG_IGN;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction (SIGPIPE, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  if (pipe (fd) != 0)
    {
      puts ("pipe failed");
      return 1;
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      return 1;
    }

  /* This will cause the read in the child to return.  */
  close (fd[0]);

  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (r != PTHREAD_CANCELED)
    {
      puts ("result is wrong");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
