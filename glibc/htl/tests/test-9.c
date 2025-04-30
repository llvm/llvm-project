/* Test recursive mutexes.
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
#include <error.h>
#include <errno.h>

#define THREADS 10

int foo;

void *
thr (void *arg)
{
  int i;

  pthread_mutex_lock (arg);

  foo = pthread_self ();

  for (i = 0; i < 500; i++)
    pthread_mutex_lock (arg);
  for (i = 0; i < 500; i++)
    pthread_mutex_unlock (arg);

  assert (foo == pthread_self ());

  pthread_mutex_unlock (arg);

  return 0;
}

int
main (int argc, char **argv)
{
  error_t err;
  int i;
  pthread_t tid[THREADS];
  pthread_mutexattr_t mattr;
  pthread_mutex_t mutex;

  err = pthread_mutexattr_init (&mattr);
  if (err)
    error (1, err, "pthread_mutexattr_init");

  err = pthread_mutexattr_settype (&mattr, PTHREAD_MUTEX_RECURSIVE);
  if (err)
    error (1, err, "pthread_mutexattr_settype");

  err = pthread_mutex_init (&mutex, &mattr);
  if (err)
    error (1, err, "pthread_mutex_init");

  err = pthread_mutexattr_destroy (&mattr);
  if (err)
    error (1, err, "pthread_mutexattr_destroy");

  pthread_mutex_lock (&mutex);
  pthread_mutex_lock (&mutex);
  pthread_mutex_unlock (&mutex);
  pthread_mutex_unlock (&mutex);

  for (i = 0; i < THREADS; i++)
    {
      err = pthread_create (&tid[i], 0, thr, &mutex);
      if (err)
	error (1, err, "pthread_create (%d)", i);
    }

  for (i = 0; i < THREADS; i++)
    {
      void *ret;

      err = pthread_join (tid[i], &ret);
      if (err)
	error (1, err, "pthread_join");

      assert (ret == 0);
    }

  err = pthread_mutex_destroy (&mutex);
  if (err)
    error (1, err, "pthread_mutex_destroy");

  return 0;
}
