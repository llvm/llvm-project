/* Helper program for testing the pthread_mutex_t and pthread_mutexattr_t
   pretty printers.

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

#include <pthread.h>

#define PASS 0
#define FAIL 1
#define PRIOCEILING 42

/* Need these so we don't have lines longer than 79 chars.  */
#define SET_TYPE(attr, type) pthread_mutexattr_settype (attr, type)
#define SET_ROBUST(attr, robust) pthread_mutexattr_setrobust (attr, robust)
#define SET_SHARED(attr, shared) pthread_mutexattr_setpshared (attr, shared)
#define SET_PROTOCOL(attr, protocol) \
	pthread_mutexattr_setprotocol (attr, protocol)
#define SET_PRIOCEILING(mutex, prioceiling, old_ceiling) \
	pthread_mutex_setprioceiling (mutex, prioceiling, old_ceiling)

static int mutex_reinit (pthread_mutex_t *mutex,
			 const pthread_mutexattr_t *attr);
static int test_settype (pthread_mutex_t *mutex, pthread_mutexattr_t *attr);
static int test_setrobust (pthread_mutex_t *mutex, pthread_mutexattr_t *attr);
static int test_setpshared (pthread_mutex_t *mutex, pthread_mutexattr_t *attr);
static int test_setprotocol (pthread_mutex_t *mutex,
			     pthread_mutexattr_t *attr);

int
main (void)
{
  pthread_mutex_t mutex;
  pthread_mutexattr_t attr;
  int result = FAIL;

  if (pthread_mutexattr_init (&attr) == 0
      && pthread_mutex_init (&mutex, NULL) == 0
      && test_settype (&mutex, &attr) == PASS
      && test_setrobust (&mutex, &attr) == PASS
      && test_setpshared (&mutex, &attr) == PASS
      && test_setprotocol (&mutex, &attr) == PASS)
    result = PASS;
  /* Else, one of the pthread_mutex* functions failed.  */

  return result;
}

/* Destroys MUTEX and re-initializes it using ATTR.  */
static int
mutex_reinit (pthread_mutex_t *mutex, const pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (pthread_mutex_destroy (mutex) == 0
      && pthread_mutex_init (mutex, attr) == 0)
    result = PASS;

  return result;
}

/* Tests setting the mutex type.  */
static int
test_settype (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (SET_TYPE (attr, PTHREAD_MUTEX_ERRORCHECK) == 0 /* Set type.  */
      && mutex_reinit (mutex, attr) == 0
      && SET_TYPE (attr, PTHREAD_MUTEX_RECURSIVE) == 0
      && mutex_reinit (mutex, attr) == 0
      && SET_TYPE (attr, PTHREAD_MUTEX_NORMAL) == 0
      && mutex_reinit (mutex, attr) == 0)
    result = PASS;

  return result;
}

/* Tests setting whether the mutex is robust.  */
static int
test_setrobust (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (SET_ROBUST (attr, PTHREAD_MUTEX_ROBUST) == 0 /* Set robust.  */
      && mutex_reinit (mutex, attr) == 0
      && SET_ROBUST (attr, PTHREAD_MUTEX_STALLED) == 0
      && mutex_reinit (mutex, attr) == 0)
    result = PASS;

  return result;
}

/* Tests setting whether the mutex can be shared between processes.  */
static int
test_setpshared (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;

  if (SET_SHARED (attr, PTHREAD_PROCESS_SHARED) == 0 /* Set shared.  */
      && mutex_reinit (mutex, attr) == 0
      && SET_SHARED (attr, PTHREAD_PROCESS_PRIVATE) == 0
      && mutex_reinit (mutex, attr) == 0)
    result = PASS;

  return result;
}

/* Tests setting the mutex protocol and, for Priority Protect, the Priority
   Ceiling.  */
static int
test_setprotocol (pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
  int result = FAIL;
  int old_prioceiling;

  if (SET_PROTOCOL (attr, PTHREAD_PRIO_INHERIT) == 0 /* Set protocol.  */
      && mutex_reinit (mutex, attr) == 0
      && SET_PROTOCOL (attr, PTHREAD_PRIO_PROTECT) == 0
      && mutex_reinit (mutex, attr) == 0
      && SET_PRIOCEILING(mutex, PRIOCEILING, &old_prioceiling) == 0
      && SET_PROTOCOL (attr, PTHREAD_PRIO_NONE) == 0
      && mutex_reinit (mutex, attr) == 0)
    result = PASS;

  return result;
}
