/* Compatibility routines for cthreads.
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

#include <assert.h>
#include <pthreadP.h>

#define	CTHREAD_KEY_INVALID (__cthread_key_t) -1

void
__cthread_detach (__cthread_t thread)
{
  int err;

  err = __pthread_detach ((pthread_t) thread);
  assert_perror (err);
}
weak_alias (__cthread_detach, cthread_detach)

__cthread_t
__cthread_fork (__cthread_fn_t func, void *arg)
{
  pthread_t thread;
  int err;

  err = __pthread_create (&thread, NULL, func, arg);
  assert_perror (err);

  return (__cthread_t) thread;
}
weak_alias (__cthread_fork, cthread_fork)

int
__cthread_keycreate (__cthread_key_t *key)
{
  error_t err;

  err = __pthread_key_create (key, 0);
  if (err)
    {
      errno = err;
      *key = CTHREAD_KEY_INVALID;
      err = -1;
    }

  return err;
}
weak_alias (__cthread_keycreate, cthread_keycreate)

int
__cthread_getspecific (__cthread_key_t key, void **val)
{
  *val = __pthread_getspecific (key);
  return 0;
}
weak_alias (__cthread_getspecific, cthread_getspecific)

int
__cthread_setspecific (__cthread_key_t key, void *val)
{
  error_t err;

  err = __pthread_setspecific (key, (const void *) val);
  if (err)
    {
      errno = err;
      err = -1;
    }

  return err;
}
weak_alias (__cthread_setspecific, cthread_setspecific)

void
__mutex_lock_solid (void *lock)
{
  __pthread_mutex_lock (lock);
}

void
__mutex_unlock_solid (void *lock)
{
  if (__pthread_spin_trylock (lock) != 0)
    /* Somebody already got the lock, that one will manage waking up others */
    return;
  __pthread_mutex_unlock (lock);
}
