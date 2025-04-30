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


static int
do_test (void)
{
  pthread_rwlock_t r;

  if (pthread_rwlock_init (&r, NULL) != 0)
    {
      puts ("rwlock_init failed");
      return 1;
    }
  puts ("rwlock_init succeeded");

  if (pthread_rwlock_rdlock (&r) != 0)
    {
      puts ("1st rwlock_rdlock failed");
      return 1;
    }
  puts ("1st rwlock_rdlock succeeded");

  if (pthread_rwlock_rdlock (&r) != 0)
    {
      puts ("2nd rwlock_rdlock failed");
      return 1;
    }
  puts ("2nd rwlock_rdlock succeeded");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("1st rwlock_unlock failed");
      return 1;
    }
  puts ("1st rwlock_unlock succeeded");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("2nd rwlock_unlock failed");
      return 1;
    }
  puts ("2nd rwlock_unlock succeeded");

  if (pthread_rwlock_wrlock (&r) != 0)
    {
      puts ("1st rwlock_wrlock failed");
      return 1;
    }
  puts ("1st rwlock_wrlock succeeded");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("3rd rwlock_unlock failed");
      return 1;
    }
  puts ("3rd rwlock_unlock succeeded");

  if (pthread_rwlock_wrlock (&r) != 0)
    {
      puts ("2nd rwlock_wrlock failed");
      return 1;
    }
  puts ("2nd rwlock_wrlock succeeded");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("4th rwlock_unlock failed");
      return 1;
    }
  puts ("4th rwlock_unlock succeeded");

  if (pthread_rwlock_rdlock (&r) != 0)
    {
      puts ("3rd rwlock_rdlock failed");
      return 1;
    }
  puts ("3rd rwlock_rdlock succeeded");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("5th rwlock_unlock failed");
      return 1;
    }
  puts ("5th rwlock_unlock succeeded");

  if (pthread_rwlock_destroy (&r) != 0)
    {
      puts ("rwlock_destroy failed");
      return 1;
    }
  puts ("rwlock_destroy succeeded");

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
