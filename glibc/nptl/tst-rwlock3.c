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

/* This test case checks more than standard compliance.  An
   implementation may provide this service but it is not required to
   do so.  */

#include <errno.h>
#include <pthread.h>
#include <stdio.h>


static int
do_test (void)
{
  pthread_rwlock_t r;
  int e;

  if (pthread_rwlock_init (&r, NULL) != 0)
    {
      puts ("rwlock_init failed");
      return 1;
    }
  puts ("rwlock_init succeeded");

  if (pthread_rwlock_trywrlock (&r) != 0)
    {
      puts ("rwlock_trywrlock on unlocked rwlock failed");
      return 1;
    }
  puts ("rwlock_trywrlock on unlocked rwlock succeeded");

  e = pthread_rwlock_rdlock (&r);
  if (e == 0)
    {
      puts ("rwlock_rdlock on rwlock with writer succeeded");
      return 1;
    }
  if (e != EDEADLK)
    {
      puts ("rwlock_rdlock on rwlock with writer failed != EDEADLK");
      return 1;
    }
  puts ("rwlock_rdlock on rwlock with writer failed with EDEADLK");

  e = pthread_rwlock_wrlock (&r);
  if (e == 0)
    {
      puts ("rwlock_wrlock on rwlock with writer succeeded");
      return 1;
    }
  if (e != EDEADLK)
    {
      puts ("rwlock_wrlock on rwlock with writer failed != EDEADLK");
      return 1;
    }
  puts ("rwlock_wrlock on rwlock with writer failed with EDEADLK");

  if (pthread_rwlock_unlock (&r) != 0)
    {
      puts ("rwlock_unlock failed");
      return 1;
    }
  puts ("rwlock_unlock succeeded");

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
