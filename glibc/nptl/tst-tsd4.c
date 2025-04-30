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

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static pthread_key_t key;


static int rounds;


static void
destr (void *arg)
{
  ++rounds;

  /* Use an arbirary but valid pointer to avoid GCC warnings.  */
  if (pthread_setspecific (key, (void *) &rounds) != 0)
    {
      puts ("destr: setspecific failed");
      exit (1);
    }
}


static void *
tf (void *arg)
{
  /* Use an arbirary but valid pointer to avoid GCC warnings.  */
  if (pthread_setspecific (key, (void *) &rounds) != 0)
    {
      puts ("tf: setspecific failed");
      exit (1);
    }

  return NULL;
}


/* This test check non-standard behavior.  The standard does not
   require that the implementation has to stop calling TSD destructors
   when they are set over and over again.  But NPTL does.  */
static int
do_test (void)
{
  /* Allocate two keys, both with destructors.  */
  if (pthread_key_create (&key, destr) != 0)
    {
      puts ("key_create failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  if (pthread_join (th, NULL) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (rounds < PTHREAD_DESTRUCTOR_ITERATIONS)
    {
      printf ("rounds == %d, PTHREAD_DESTRUCTOR_ITERATIONS = %d\n",
	      rounds, PTHREAD_DESTRUCTOR_ITERATIONS);
      return 1;
    }

  if (pthread_getspecific (key) != NULL)
    {
      puts ("key data != NULL");
      return 1;
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
