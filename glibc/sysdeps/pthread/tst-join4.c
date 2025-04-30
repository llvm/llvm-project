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
#include <stdio.h>
#include <stdlib.h>


static pthread_barrier_t bar;


static void *
tf (void *arg)
{
  if (pthread_barrier_wait (&bar) != 0)
    {
      puts ("tf: barrier_wait failed");
      exit (1);
    }

  return (void *) 1l;
}


static int
do_test (void)
{
  if (pthread_barrier_init (&bar, NULL, 3) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  pthread_attr_t a;

  if (pthread_attr_init (&a) != 0)
    {
      puts ("attr_init failed");
      exit (1);
    }

  if (pthread_attr_setstacksize (&a, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

  pthread_t th[2];

  if (pthread_create (&th[0], &a, tf, NULL) != 0)
    {
      puts ("1st create failed");
      exit (1);
    }

  if (pthread_attr_setdetachstate (&a, PTHREAD_CREATE_DETACHED) != 0)
    {
      puts ("attr_setdetachstate failed");
      exit (1);
    }

  if (pthread_create (&th[1], &a, tf, NULL) != 0)
    {
      puts ("1st create failed");
      exit (1);
    }

  if (pthread_attr_destroy (&a) != 0)
    {
      puts ("attr_destroy failed");
      exit (1);
    }

  if (pthread_detach (th[0]) != 0)
    {
      puts ("could not detach 1st thread");
      exit (1);
    }

  int err = pthread_detach (th[0]);
  if (err == 0)
    {
      puts ("second detach of 1st thread succeeded");
      exit (1);
    }
  if (err != EINVAL)
    {
      printf ("second detach of 1st thread returned %d, not EINVAL\n", err);
      exit (1);
    }

  err = pthread_detach (th[1]);
  if (err == 0)
    {
      puts ("detach of 2nd thread succeeded");
      exit (1);
    }
  if (err != EINVAL)
    {
      printf ("detach of 2nd thread returned %d, not EINVAL\n", err);
      exit (1);
    }

  exit (0);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
