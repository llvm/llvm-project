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
#include <stdlib.h>
#include <unistd.h>


static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static int global;


static void
ch (void *arg)
{
  int val = (long int) arg;

  printf ("ch (%d)\n", val);

  global *= val;
  global += val;
}


static void *
tf (void *a)
{
  pthread_cancel (pthread_self ());

  pthread_cleanup_push (ch, (void *) 1l);

  pthread_cleanup_push (ch, (void *) 2l);

  pthread_cleanup_push (ch, (void *) 3l);

  pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

  pthread_cleanup_pop (1);

  pthread_cleanup_pop (1);

  pthread_cleanup_pop (1);

  return NULL;
}


int
do_test (void)
{
  pthread_t th;

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      write_message ("create failed\n");
      _exit (1);
    }

  void *r;
  int e;
  if ((e = pthread_join (th, &r)) != 0)
    {
      printf ("join failed: %d\n", e);
      _exit (1);
    }

  if (r != PTHREAD_CANCELED)
    {
      puts ("thread not canceled");
      exit (1);
    }

  if (global != 9)
    {
      printf ("global = %d, expected 9\n", global);
      exit (1);
    }

  return 0;
}
