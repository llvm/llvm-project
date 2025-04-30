/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <unistd.h>
#include <sys/types.h>


static int global;

static void *
tf (void *a)
{
  global = 1;

  return 0;
}


int
do_test (void)
{
  pthread_t th;
  pthread_attr_t at;

  if (pthread_attr_init (&at) != 0)
    {
      puts ("attr_init failed");
      return 1;
    }

  if (pthread_attr_setschedpolicy (&at, SCHED_OTHER) != 0)
    {
      puts ("attr_setschedpolicy failed");
      return 1;
    }

  struct sched_param pa;
  if (sched_getparam (getpid (), &pa) != 0)
    {
      puts ("sched_getschedparam failed");
      return 1;
    }

  if (pthread_attr_setschedparam (&at, &pa) != 0)
    {
      puts ("attr_setschedparam failed");
      return 1;
    }

  if (pthread_attr_setinheritsched (&at, PTHREAD_EXPLICIT_SCHED) != 0)
    {
      puts ("attr_setinheritsched failed");
      return 1;
    }

  if (pthread_create (&th, &at, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  int e = pthread_join (th, NULL);
  if (e != 0)
    {
      printf ("join failed: %d\n", e);
      return 1;
    }

  if (global == 0)
    {
      puts ("thread didn't run");
      return 1;
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
