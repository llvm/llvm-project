/* Test pthread_rwlock_timedrdlock and pthread_rwlock_timedwrlock.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#define _GNU_SOURCE

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <error.h>
#include <errno.h>
#include <sys/time.h>

#define THREADS 10

pthread_rwlock_t rwlock;

void *
test (void *arg)
{
  error_t err;
  int foo = (int) arg;
  struct timespec ts;
  struct timeval before, after;
  int diff;

  gettimeofday (&before, 0);
  ts.tv_sec = before.tv_sec + 1;
  ts.tv_nsec = before.tv_usec * 1000;

  printf ("Thread %d starting wait @ %d\n", pthread_self (),
	  (int) before.tv_sec);

  if (foo % 2 == 0)
    err = pthread_rwlock_timedrdlock (&rwlock, &ts);
  else
    err = pthread_rwlock_timedwrlock (&rwlock, &ts);

  assert (err == ETIMEDOUT);

  gettimeofday (&after, 0);

  printf ("Thread %d ending wait @ %d\n", pthread_self (), (int) after.tv_sec);

  diff = after.tv_sec * 1000000 + after.tv_usec
      - before.tv_sec * 1000000 - before.tv_usec;

  if (diff < 900000 || diff > 1100000)
    error (1, EGRATUITOUS, "pthread_mutex_timedlock waited %d us", diff);

  return 0;
}

int
main (int argc, char **argv)
{
  error_t err;
  int i;
  pthread_t tid[THREADS];

  err = pthread_rwlock_init (&rwlock, 0);
  if (err)
    error (1, err, "pthread_rwlock_init");

  /* Lock it so all the threads will block.  */
  err = pthread_rwlock_wrlock (&rwlock);
  assert (err == 0);

  for (i = 0; i < THREADS; i++)
    {
      err = pthread_create (&tid[i], 0, test, (void *) i);
      if (err)
	error (1, err, "pthread_create");
    }

  for (i = 0; i < THREADS; i++)
    {
      void *ret;

      err = pthread_join (tid[i], &ret);
      if (err)
	error (1, err, "pthread_join");

      assert (ret == 0);
    }

  return 0;
}
