/* Helper program for testing the pthread_rwlock_t and pthread_rwlockattr_t
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

/* Need these so we don't have lines longer than 79 chars.  */
#define SET_KIND(attr, kind) pthread_rwlockattr_setkind_np (attr, kind)
#define SET_SHARED(attr, shared) pthread_rwlockattr_setpshared (attr, shared)

static int rwlock_reinit (pthread_rwlock_t *rwlock,
			  const pthread_rwlockattr_t *attr);
static int test_setkind_np (pthread_rwlock_t *rwlock,
			    pthread_rwlockattr_t *attr);
static int test_setpshared (pthread_rwlock_t *rwlock,
			    pthread_rwlockattr_t *attr);

int
main (void)
{
  pthread_rwlock_t rwlock;
  pthread_rwlockattr_t attr;
  int result = FAIL;

  if (pthread_rwlockattr_init (&attr) == 0
      && pthread_rwlock_init (&rwlock, NULL) == 0
      && test_setkind_np (&rwlock, &attr) == PASS
      && test_setpshared (&rwlock, &attr) == PASS)
    result = PASS;
  /* Else, one of the pthread_rwlock* functions failed.  */

  return result;
}

/* Destroys RWLOCK and re-initializes it using ATTR.  */
static int
rwlock_reinit (pthread_rwlock_t *rwlock, const pthread_rwlockattr_t *attr)
{
  int result = FAIL;

  if (pthread_rwlock_destroy (rwlock) == 0
      && pthread_rwlock_init (rwlock, attr) == 0)
    result = PASS;

  return result;
}

/* Tests setting whether the rwlock prefers readers or writers.  */
static int
test_setkind_np (pthread_rwlock_t *rwlock, pthread_rwlockattr_t *attr)
{
  int result = FAIL;

  if (SET_KIND (attr, PTHREAD_RWLOCK_PREFER_READER_NP) == 0 /* Set kind.  */
      && rwlock_reinit (rwlock, attr) == PASS
      && SET_KIND (attr, PTHREAD_RWLOCK_PREFER_WRITER_NP) == 0
      && rwlock_reinit (rwlock, attr) == PASS
      && SET_KIND (attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP) == 0
      && rwlock_reinit (rwlock, attr) == PASS)
    result = PASS;

  return result;
}

/* Tests setting whether the rwlock can be shared between processes.  */
static int
test_setpshared (pthread_rwlock_t *rwlock, pthread_rwlockattr_t *attr)
{
  int result = FAIL;

  if (SET_SHARED (attr, PTHREAD_PROCESS_SHARED) == 0 /* Set shared.  */
      && rwlock_reinit (rwlock, attr) == PASS
      && SET_SHARED (attr, PTHREAD_PROCESS_PRIVATE) == 0
      && rwlock_reinit (rwlock, attr) == PASS)
    result = PASS;

  return result;
}
