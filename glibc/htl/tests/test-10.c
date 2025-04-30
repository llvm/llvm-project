/* Test error checking mutexes.
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

int
main (int argc, char **argv)
{
  error_t err;
  pthread_mutexattr_t mattr;
  pthread_mutex_t mutex;

  err = pthread_mutexattr_init (&mattr);
  if (err)
    error (1, err, "pthread_mutexattr_init");

  err = pthread_mutexattr_settype (&mattr, PTHREAD_MUTEX_ERRORCHECK);
  if (err)
    error (1, err, "pthread_mutexattr_settype");

  err = pthread_mutex_init (&mutex, &mattr);
  if (err)
    error (1, err, "pthread_mutex_init");

  err = pthread_mutexattr_destroy (&mattr);
  if (err)
    error (1, err, "pthread_mutexattr_destroy");

  err = pthread_mutex_lock (&mutex);
  assert (err == 0);

  err = pthread_mutex_lock (&mutex);
  assert (err == EDEADLK);

  err = pthread_mutex_unlock (&mutex);
  assert (err == 0);

  err = pthread_mutex_unlock (&mutex);
  assert (err == EPERM);

  return 0;
}
