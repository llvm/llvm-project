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
  pthread_spinlock_t s;

  if (pthread_spin_init (&s, PTHREAD_PROCESS_PRIVATE) != 0)
    {
      puts ("spin_init failed");
      return 1;
    }

  if (pthread_spin_lock (&s) != 0)
    {
      puts ("spin_lock failed");
      return 1;
    }

  if (pthread_spin_unlock (&s) != 0)
    {
      puts ("spin_unlock failed");
      return 1;
    }

  if (pthread_spin_destroy (&s) != 0)
    {
      puts ("spin_destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
