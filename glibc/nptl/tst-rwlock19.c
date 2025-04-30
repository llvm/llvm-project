/* Test rdlock overflow.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <error.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <pthreadP.h>


#define NREADERS 15
#define READTRIES 5000

#define DELAY   1000000

static pthread_rwlock_t lock = PTHREAD_RWLOCK_INITIALIZER;
static int eagain_returned = 0;
static int success_returned = 0;

static void *
reader_thread (void *nr)
{
  struct timespec delay;
  int n;

  delay.tv_sec = 0;
  delay.tv_nsec = DELAY;

  for (n = 0; n < READTRIES; ++n)
    {
      int err = pthread_rwlock_rdlock (&lock);
      if (err == EAGAIN)
	{
	  atomic_store_relaxed (&eagain_returned, 1);
	  continue;
	}
      else if (err == 0)
	atomic_store_relaxed (&success_returned, 1);
      else
	{
	  puts ("rdlock failed");
	  exit (1);
	}

      nanosleep (&delay, NULL);

      if (pthread_rwlock_unlock (&lock) != 0)
	{
	  puts ("unlock for reader failed");
	  exit (1);
	}
    }

  return NULL;
}


static int
do_test (void)
{
  pthread_t thrd[NREADERS];
  int n;
  void *res;

  /* Set the rwlock so that it's close to a reader overflow.
     PTHREAD_RWLOCK_WRPHASE and PTHREAD_RWLOCK_WRLOCK are zero initially.  */
  unsigned int readers = PTHREAD_RWLOCK_READER_OVERFLOW
      - ((NREADERS / 3) << PTHREAD_RWLOCK_READER_SHIFT);
  lock.__data.__readers = readers;

  for (n = 0; n < NREADERS; ++n)
    if (pthread_create (&thrd[n], NULL, reader_thread,
			(void *) (long int) n) != 0)
      {
	puts ("reader create failed");
	exit (1);
      }

  /* Wait for all the threads.  */
  for (n = 0; n < NREADERS; ++n)
    if (pthread_join (thrd[n], &res) != 0)
      {
	puts ("reader join failed");
	exit (1);
      }

  if (atomic_load_relaxed (&eagain_returned) == 0)
    {
      puts ("EAGAIN has never been returned");
      exit (1);
    }

  if (atomic_load_relaxed (&success_returned) == 0)
    {
      puts ("rdlock was never successfully acquired");
      exit (1);
    }

  if (lock.__data.__readers != readers)
    {
      puts ("__readers in rwlock differs from initial value");
      exit (1);
    }

  return 0;
}

#define TIMEOUT 100
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
