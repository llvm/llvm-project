/* Test rwlocks.
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

#define THREADS 1

int a;
int b;

/* Get a read lock and assert that a == b.  */
void *
test1 (void *arg)
{
  error_t err;
  pthread_rwlock_t *lock = arg;
  int i;

  for (i = 0; i < 200; i++)
    {
      err = pthread_rwlock_rdlock (lock);
      assert (err == 0);

      assert (a == b);

      sched_yield ();

      assert (a == b);

      err = pthread_rwlock_unlock (lock);
      assert (err == 0);
    }

  return 0;
}

int
main (int argc, char **argv)
{
  error_t err;
  pthread_rwlockattr_t attr;
  pthread_rwlock_t lock;
  int pshared;

  int i;
  pthread_t tid[THREADS];
  void *ret;

  err = pthread_rwlockattr_init (&attr);
  if (err)
    error (1, err, "pthread_rwlockattr_init");

  err = pthread_rwlockattr_getpshared (&attr, &pshared);
  if (err)
    error (1, err, "pthread_rwlockattr_getpshared");

  /* Assert the default state as mandated by POSIX.  */
  assert (pshared == PTHREAD_PROCESS_PRIVATE);

  err = pthread_rwlockattr_setpshared (&attr, pshared);
  if (err)
    error (1, err, "pthread_rwlockattr_setpshared");

  err = pthread_rwlock_init (&lock, &attr);
  if (err)
    error (1, err, "pthread_rwlock_init");

  err = pthread_rwlockattr_destroy (&attr);
  if (err)
    error (1, err, "pthread_rwlockattr_destroy");

  /* Now test the lock.  */

  for (i = 0; i < THREADS; i++)
    {
      err = pthread_create (&tid[i], 0, test1, &lock);
      if (err)
	error (1, err, "pthread_create");
    }

  for (i = 0; i < 10; i++)
    {
      sched_yield ();

      /* Get a write lock.  */
      pthread_rwlock_wrlock (&lock);
      /* Increment a and b giving other threads a chance to run in
         between.  */
      sched_yield ();
      a++;
      sched_yield ();
      b++;
      sched_yield ();
      /* Unlock.  */
      pthread_rwlock_unlock (&lock);
    }

  for (i = 0; i < THREADS; i++)
    {
      err = pthread_join (tid[i], &ret);
      if (err)
	error (1, err, "pthread_join");
    }

  /* Read lock it.  */
  err = pthread_rwlock_tryrdlock (&lock);
  assert (err == 0);

  /* Try to write lock it.  It should fail with EBUSY.  */
  err = pthread_rwlock_trywrlock (&lock);
  assert (err == EBUSY);

  /* Drop the read lock.  */
  err = pthread_rwlock_unlock (&lock);
  assert (err == 0);

  /* Get a write lock.  */
  err = pthread_rwlock_trywrlock (&lock);
  assert (err == 0);

  /* Fail trying to acquire another write lock.  */
  err = pthread_rwlock_trywrlock (&lock);
  assert (err == EBUSY);

  /* Try to get a read lock which should also fail.  */
  err = pthread_rwlock_tryrdlock (&lock);
  assert (err == EBUSY);

  /* Unlock it.  */
  err = pthread_rwlock_unlock (&lock);
  assert (err == 0);


  err = pthread_rwlock_destroy (&lock);
  if (err)
    error (1, err, "pthread_rwlock_destroy");

  return 0;
}
