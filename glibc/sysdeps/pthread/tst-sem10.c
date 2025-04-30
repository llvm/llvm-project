/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2007.

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
#include <semaphore.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>


static int
do_test (void)
{
  sem_t s;
  if (sem_init (&s, 0, 0) == -1)
    {
      puts ("sem_init failed");
      return 1;
    }

  struct timeval tv;
  if (gettimeofday (&tv, NULL) != 0)
    {
      puts ("gettimeofday failed");
      return 1;
    }

  struct timespec ts;
  TIMEVAL_TO_TIMESPEC (&tv, &ts);

  /* Set ts to yesterday.  */
  ts.tv_sec -= 86400;

  int type_before;
  if (pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &type_before) != 0)
    {
      puts ("first pthread_setcanceltype failed");
      return 1;
    }

  errno = 0;
  if (TEMP_FAILURE_RETRY (sem_timedwait (&s, &ts)) != -1)
    {
      puts ("sem_timedwait succeeded");
      return 1;
    }
  if (errno != ETIMEDOUT)
    {
      printf ("sem_timedwait return errno = %d instead of ETIMEDOUT\n",
	      errno);
      return 1;
    }

  int type_after;
  if (pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &type_after) != 0)
    {
      puts ("second pthread_setcanceltype failed");
      return 1;
    }
  if (type_after != PTHREAD_CANCEL_DEFERRED)
    {
      puts ("sem_timedwait changed cancellation type");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
