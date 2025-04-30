/* Helper program for testing the pthread_mutex_t pretty printer.

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Keep the calls to the pthread_* functions on separate lines to make it easy
   to advance through the program using the gdb 'next' command.  */

#include <stdlib.h>
#include <errno.h>
#include <pthread.h>

#define PASS 0
#define FAIL 1

static int test_status_destroyed (pthread_mutex_t *mutex);
static int test_status_no_robust (pthread_mutex_t *mutex,
				  pthread_mutexattr_t *attr);
static int test_status_robust (pthread_mutex_t *mutex,
			       pthread_mutexattr_t *attr);
static int test_locking_state_robust (pthread_mutex_t *mutex);
static void *thread_func (void *arg);
static int test_recursive_locks (pthread_mutex_t *mutex,
				 pthread_mutexattr_t *attr);

int
main (void)
{
  pthread_mutex_t mutex;
  pthread_mutexattr_t attr;
  int result = FAIL;

  if (pthread_mutexattr_init (&attr) == 0
      && test_status_destroyed (&mutex) == PASS
      && test_status_no_robust (&mutex, &attr) == PASS
      && test_status_robust (&mutex, &attr) == PASS
      && test_recursive_locks (&mutex, &attr) == PASS)
    result = PASS;
  /* Else, one of the pthread_mutex* functions failed.  */

  return result;
}

/* Initializes MUTEX, then destroys it.  */
static int
test_status_destroyed (pthread_mutex_t *mutex)
{
  int result = FAIL;

  if (pthread_mutex_init (mutex, NULL) == 0
      && pthread_mutex_destroy (mutex) == 0)
    result = PASS; /* Test status (destroyed).  */

  return result;
}

/* Tests locking of non-robust mutexes.  */
static int
test_status_no_robust (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (pthread_mutexattr_setrobust (attr, PTHREAD_MUTEX_STALLED) == 0
      && pthread_mutex_init (mutex, attr) == 0
      && pthread_mutex_lock (mutex) == 0 /* Test status (non-robust).  */
      && pthread_mutex_unlock (mutex) == 0
      && pthread_mutex_destroy (mutex) == 0)
    result = PASS;

  return result;
}

/* Tests locking of robust mutexes.  */
static int
test_status_robust (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (pthread_mutexattr_setrobust (attr, PTHREAD_MUTEX_ROBUST) == 0
      && pthread_mutex_init (mutex, attr) == 0
      && test_locking_state_robust (mutex) == PASS /* Test status (robust).  */
      && pthread_mutex_destroy (mutex) == 0)
    result = PASS;

  return result;
}

/* Tests locking and state corruption of robust mutexes.  We'll mark it as
   inconsistent, then not recoverable.  */
static int
test_locking_state_robust (pthread_mutex_t *mutex)
{
  int result = FAIL;
  pthread_t thread;

  if (pthread_create (&thread, NULL, thread_func, mutex) == 0 /* Create.  */
      && pthread_join (thread, NULL) == 0
      && pthread_mutex_lock (mutex) == EOWNERDEAD /* Test locking (robust).  */
      && pthread_mutex_unlock (mutex) == 0)
    result = PASS;

  return result;
}

/* Function to be called by the child thread when testing robust mutexes.  */
static void *
thread_func (void *arg)
{
  pthread_mutex_t *mutex = (pthread_mutex_t *)arg;

  if (pthread_mutex_lock (mutex) != 0) /* Thread function.  */
    exit (FAIL);

  /* Thread terminates without unlocking the mutex, thus marking it as
     inconsistent.  */
  return NULL;
}

/* Tests locking the mutex multiple times in a row.  */
static int
test_recursive_locks (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (pthread_mutexattr_settype (attr, PTHREAD_MUTEX_RECURSIVE) == 0
      && pthread_mutex_init (mutex, attr) == 0
      && pthread_mutex_lock (mutex) == 0
      && pthread_mutex_lock (mutex) == 0
      && pthread_mutex_lock (mutex) == 0 /* Test recursive locks.  */
      && pthread_mutex_unlock (mutex) == 0
      && pthread_mutex_unlock (mutex) == 0
      && pthread_mutex_unlock (mutex) == 0
      && pthread_mutex_destroy (mutex) == 0)
    result = PASS;

  return result;
}
