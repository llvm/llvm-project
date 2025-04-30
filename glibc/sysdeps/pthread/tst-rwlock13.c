/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <stdio.h>
#include <string.h>


static int
do_test (void)
{
  pthread_rwlock_t r;
  int ret;

  memset (&r, 0xaa, sizeof (r));
  if ((ret = pthread_rwlock_init (&r, NULL)) != 0)
    {
      printf ("rwlock_init failed: %d\n", ret);
      return 1;
    }

  if ((ret = pthread_rwlock_rdlock (&r)) != 0)
    {
      printf ("rwlock_rdlock failed: %d\n", ret);
      return 1;
    }

  if ((ret = pthread_rwlock_unlock (&r)) != 0)
    {
      printf ("rwlock_unlock failed: %d\n", ret);
      return 1;
    }

  if ((ret = pthread_rwlock_wrlock (&r)) != 0)
    {
      printf ("rwlock_wrlock failed: %d\n", ret);
      return 1;
    }

  if ((ret = pthread_rwlock_unlock (&r)) != 0)
    {
      printf ("second rwlock_unlock failed: %d\n", ret);
      return 1;
    }

  if ((ret = pthread_rwlock_destroy (&r)) != 0)
    {
      printf ("second rwlock_destroy failed: %d\n", ret);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
