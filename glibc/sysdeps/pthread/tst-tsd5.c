/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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


static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;


static void
cl (void *p)
{
  pthread_mutex_unlock (&m);
}


static void *
tf (void *arg)
{
  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("2nd mutex_lock failed");
      exit (1);
    }

  exit (0);
}


static int
do_test (void)
{
  pthread_key_t k;
  if (pthread_key_create (&k, cl) != 0)
    {
      puts ("key_create failed");
      return 1;
    }
  /* Use an arbitrary but valid pointer as the value.  */
  if (pthread_setspecific (k, (void *) &k) != 0)
    {
      puts ("setspecific failed");
      return 1;
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("1st mutex_lock failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  pthread_exit (NULL);
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
