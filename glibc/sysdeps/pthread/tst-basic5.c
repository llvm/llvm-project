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



int
do_test (void)
{
  int c = pthread_getconcurrency ();
  if (c != 0)
    {
      puts ("initial concurrencylevel wrong");
      exit (1);
    }

  if (pthread_setconcurrency (1) != 0)
    {
      puts ("setconcurrency failed");
      exit (1);
    }

  c = pthread_getconcurrency ();
  if (c != 1)
    {
      puts ("getconcurrency didn't return the value previous set");
      exit (1);
    }

  int e = pthread_setconcurrency (-1);
  if (e == 0)
    {
      puts ("setconcurrency of negative value didn't failed");
      exit (1);
    }
  if (e != EINVAL)
    {
      puts ("setconcurrency didn't return EINVAL for negative value");
      exit (1);
    }

  c = pthread_getconcurrency ();
  if (c != 1)
    {
      puts ("invalid getconcurrency changed level");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
