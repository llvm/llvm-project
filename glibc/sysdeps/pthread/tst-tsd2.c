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
#include <string.h>


static int result;


static void
destr (void *arg)
{
  if (arg != (void *) &result)
    result = 2;
  else
    result = 0;
}


static void *
tf (void *arg)
{
  pthread_key_t key = (pthread_key_t) (long int) arg;
  int err;

  /* Use an arbirary but valid pointer to avoid GCC warnings.  */
  err = pthread_setspecific (key, &result);
  if (err != 0)
    result = 3;

  return NULL;
}


static int
do_test (void)
{
  pthread_key_t key;
  pthread_t th;
  int err;

  err = pthread_key_create (&key, destr);
  if (err != 0)
    {
      printf ("key_create failed: %s\n", strerror (err));
      return 1;
    }

  result = 1;

  err = pthread_create (&th, NULL, tf, (void *) (long int) key);
  if (err != 0)
    {
      printf ("create failed: %s\n", strerror (err));
      return 1;
    }

  /* Wait for the thread to terminate.  */
  err = pthread_join (th, NULL);
  if (err != 0)
    {
      printf ("join failed: %s\n", strerror (err));
      return 1;
    }

  if (result == 1)
    puts ("destructor not called");
  else if (result == 2)
    puts ("destructor got passed a wrong value");
  else if (result == 3)
    puts ("setspecific in child failed");
  else if (result != 0)
    puts ("result != 0");

  return result;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
