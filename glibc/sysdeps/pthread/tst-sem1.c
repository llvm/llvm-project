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
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>


static int
do_test (void)
{
  sem_t s;

  if (sem_init (&s, 0, 1) == -1)
    {
      puts ("init failed");
      return 1;
    }

  if (TEMP_FAILURE_RETRY (sem_wait (&s)) == -1)
    {
      puts ("1st wait failed");
      return 1;
    }

  if (sem_post (&s) == -1)
    {
      puts ("1st post failed");
      return 1;
    }

  if (TEMP_FAILURE_RETRY (sem_trywait (&s)) == -1)
    {
      puts ("1st trywait failed");
      return 1;
    }

  errno = 0;
  if (TEMP_FAILURE_RETRY (sem_trywait (&s)) != -1)
    {
      puts ("2nd trywait succeeded");
      return 1;
    }
  else if (errno != EAGAIN)
    {
      puts ("2nd trywait did not set errno to EAGAIN");
      return 1;
    }

  if (sem_post (&s) == -1)
    {
      puts ("2nd post failed");
      return 1;
    }

  if (TEMP_FAILURE_RETRY (sem_wait (&s)) == -1)
    {
      puts ("2nd wait failed");
      return 1;
    }

  if (sem_destroy (&s) == -1)
    {
      puts ("destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
