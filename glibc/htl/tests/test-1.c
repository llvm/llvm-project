/* Test mutexes.
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
#include <assert.h>
#include <unistd.h>
#include <error.h>
#include <errno.h>
#include <stdio.h>

#define THREADS 500

void *
foo (void *arg)
{
  pthread_mutex_t *mutex = arg;
  pthread_mutex_lock (mutex);
  pthread_mutex_unlock (mutex);
  return mutex;
}

int
main (int argc, char **argv)
{
  int i;
  error_t err;
  pthread_t tid[THREADS];
  pthread_mutex_t mutex[THREADS];

  for (i = 0; i < THREADS; i++)
    {
      pthread_mutex_init (&mutex[i], 0);
      pthread_mutex_lock (&mutex[i]);
      err = pthread_create (&tid[i], 0, foo, &mutex[i]);
      if (err)
	error (1, err, "pthread_create");
      sched_yield ();
    }

  for (i = THREADS - 1; i >= 0; i--)
    {
      void *ret;
      pthread_mutex_unlock (&mutex[i]);
      err = pthread_join (tid[i], &ret);
      if (err)
	error (1, err, "pthread_join");
      assert (ret == &mutex[i]);
    }

  return 0;
}
