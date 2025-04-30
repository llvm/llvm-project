/* Make sure that pthread_mutexattr_gettype returns a valid kind.

   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <stdio.h>
#include <string.h>
#include <pthread.h>

static int
do_test (void)
{
  pthread_mutexattr_t attr;
  int kind;
  int error;

  error = pthread_mutexattr_init (&attr);
  if (error)
    {
      printf ("pthread_mutexattr_init: %s\n", strerror (error));
      return 1;
    }
  error = pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_DEFAULT);
  if (error)
    {
      printf ("pthread_mutexattr_settype (1): %s\n", strerror (error));
      return 1;
    }
  error = pthread_mutexattr_gettype (&attr, &kind);
  if (error)
    {
      printf ("pthread_mutexattr_gettype: %s\n", strerror (error));
      return 1;
    }
  error = pthread_mutexattr_settype (&attr, kind);
  if (error)
    {
      printf ("pthread_mutexattr_settype (2): %s\n", strerror (error));
      return 1;
    }
  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
