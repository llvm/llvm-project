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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>


static int
do_test (void)
{
  pthread_rwlock_t r;
  pthread_rwlockattr_t at;
  int e;

  if (pthread_rwlockattr_init (&at) != 0)
    {
      puts ("rwlockattr_init failed");
      return 1;
    }
  puts ("rwlockattr_init succeeded");

#ifndef TYPE
# define TYPE PTHREAD_RWLOCK_PREFER_READER_NP
#endif

  if (pthread_rwlockattr_setkind_np (&at, TYPE) != 0)
    {
      puts ("rwlockattr_setkind failed");
      return 1;
    }
  puts ("rwlockattr_setkind succeeded");

  if (pthread_rwlock_init (&r, &at) != 0)
    {
      puts ("rwlock_init failed");
      return 1;
    }
  puts ("rwlock_init succeeded");

  if (pthread_rwlockattr_destroy (&at) != 0)
    {
      puts ("rwlockattr_destroy failed");
      return 1;
    }
  puts ("rwlockattr_destroy succeeded");

  if (pthread_rwlock_wrlock (&r) != 0)
    {
      puts ("1st rwlock_wrlock failed");
      return 1;
    }
  puts ("1st rwlock_wrlock succeeded");

  e = pthread_rwlock_tryrdlock (&r);
  if (e == 0)
    {
      puts ("rwlock_tryrdlock on rwlock with writer succeeded");
      return 1;
    }
  if (e != EBUSY)
    {
      puts ("rwlock_tryrdlock on rwlock with writer return value != EBUSY");
      return 1;
    }
  puts ("rwlock_tryrdlock on rwlock with writer failed with EBUSY");

  e = pthread_rwlock_trywrlock (&r);
  if (e == 0)
    {
      puts ("rwlock_trywrlock on rwlock with writer succeeded");
      return 1;
    }
  if (e != EBUSY)
    {
      puts ("rwlock_trywrlock on rwlock with writer return value != EBUSY");
      return 1;
    }
  puts ("rwlock_trywrlock on rwlock with writer failed with EBUSY");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("1st rwlock_unlock failed");
      return 1;
    }
  puts ("1st rwlock_unlock succeeded");

  if (pthread_rwlock_tryrdlock (&r) != 0)
    {
      puts ("rwlock_tryrdlock on unlocked rwlock failed");
      return 1;
    }
  puts ("rwlock_tryrdlock on unlocked rwlock succeeded");

  e = pthread_rwlock_trywrlock (&r);
  if (e == 0)
    {
      puts ("rwlock_trywrlock on rwlock with reader succeeded");
      return 1;
    }
  if (e != EBUSY)
    {
      puts ("rwlock_trywrlock on rwlock with reader return value != EBUSY");
      return 1;
    }
  puts ("rwlock_trywrlock on rwlock with reader failed with EBUSY");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("2nd rwlock_unlock failed");
      return 1;
    }
  puts ("2nd rwlock_unlock succeeded");

  if (pthread_rwlock_trywrlock (&r) != 0)
    {
      puts ("rwlock_trywrlock on unlocked rwlock failed");
      return 1;
    }
  puts ("rwlock_trywrlock on unlocked rwlock succeeded");

  e = pthread_rwlock_tryrdlock (&r);
  if (e == 0)
    {
      puts ("rwlock_tryrdlock on rwlock with writer succeeded");
      return 1;
    }
  if (e != EBUSY)
    {
      puts ("rwlock_tryrdlock on rwlock with writer return value != EBUSY");
      return 1;
    }
  puts ("rwlock_tryrdlock on rwlock with writer failed with EBUSY");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("3rd rwlock_unlock failed");
      return 1;
    }
  puts ("3rd rwlock_unlock succeeded");

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
