/* Test condition attributes and pthread_cond_timedwait.
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

int
main (int argc, char **argv)
{
  error_t err;
  int i;
  pthread_condattr_t attr;
  pthread_cond_t cond;
  struct timespec ts;
  pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
  struct timeval before, after;
  int diff;

  err = pthread_condattr_init (&attr);
  if (err)
    error (1, err, "pthread_condattr_init");

  err = pthread_condattr_getpshared (&attr, &i);
  if (err)
    error (1, err, "pthread_condattr_getpshared");
  assert (i == PTHREAD_PROCESS_PRIVATE);

  err = pthread_condattr_setpshared (&attr, PTHREAD_PROCESS_PRIVATE);
  assert (err == 0);

  err = pthread_cond_init (&cond, &attr);
  if (err)
    error (1, err, "pthread_cond_init");

  err = pthread_condattr_destroy (&attr);
  if (err)
    error (1, err, "pthread_condattr_destroy");

  gettimeofday (&before, 0);
  ts.tv_sec = before.tv_sec + 1;
  ts.tv_nsec = before.tv_usec * 1000;

  printf ("Starting wait @ %d\n", (int) before.tv_sec);

  pthread_mutex_lock (&m);
  err = pthread_cond_timedwait (&cond, &m, &ts);

  gettimeofday (&after, 0);

  printf ("End wait @ %d (err = %d)\n", (int) after.tv_sec, err);

  assert (err == ETIMEDOUT);

  diff = after.tv_sec * 1000000 + after.tv_usec
      - before.tv_sec * 1000000 - before.tv_usec;

  if (diff < 900000 || diff > 1100000)
    error (1, EGRATUITOUS, "pthread_cond_timedwait waited %d us", diff);

  return 0;
}
