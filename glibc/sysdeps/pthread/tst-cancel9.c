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

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static pthread_barrier_t b;


static void
cleanup (void *arg)
{
  fputs ("in cleanup\n", stdout);
}


static void *
tf (void *arg)
{
  int fd = open ("/dev/null", O_RDWR);
  if (fd == -1)
    {
      puts ("cannot open /dev/null");
      exit (1);
    }
  FILE *fp = fdopen (fd, "w");
  if (fp == NULL)
    {
      puts ("fdopen failed");
      exit (1);
    }

  pthread_cleanup_push (cleanup, NULL);

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  while (1)
    /* fprintf() uses write() which is a cancallation point.  */
    fprintf (fp, "foo");

  pthread_cleanup_pop (0);

  return NULL;
}


static int
do_test (void)
{
  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  sleep (1);

  puts ("cancel now");

  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      exit (1);
    }

  puts ("waiting for the child");

  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      exit (1);
    }

  if (r != PTHREAD_CANCELED)
    {
      puts ("thread wasn't canceled");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
