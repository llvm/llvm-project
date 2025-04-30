/* Helper program for testing the pthread_rwlock_t pretty printer.

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

static int test_locking_reader (pthread_rwlock_t *rwlock);
static int test_locking_writer (pthread_rwlock_t *rwlock);

int
main (void)
{
  pthread_rwlock_t rwlock;

  int result = FAIL;

  if (test_locking_reader (&rwlock) == PASS
      && test_locking_writer (&rwlock) == PASS)
    result = PASS;
  /* Else, one of the pthread_rwlock* functions failed.  */

  return result;
}

/* Tests locking the rwlock multiple times as a reader.  */
static int
test_locking_reader (pthread_rwlock_t *rwlock)
{
  int result = FAIL;

  if (pthread_rwlock_init (rwlock, NULL) == 0
      && pthread_rwlock_rdlock (rwlock) == 0 /* Test locking (reader).  */
      && pthread_rwlock_rdlock (rwlock) == 0
      && pthread_rwlock_rdlock (rwlock) == 0
      && pthread_rwlock_unlock (rwlock) == 0
      && pthread_rwlock_unlock (rwlock) == 0
      && pthread_rwlock_unlock (rwlock) == 0
      && pthread_rwlock_destroy (rwlock) == 0)
    result = PASS;

  return result;
}

/* Tests locking the rwlock as a writer.  */
static int
test_locking_writer (pthread_rwlock_t *rwlock)
{
  int result = FAIL;

  if (pthread_rwlock_init (rwlock, NULL) == 0
      && pthread_rwlock_wrlock (rwlock) == 0 /* Test locking (writer).  */
      && pthread_rwlock_unlock (rwlock) == 0
      && pthread_rwlock_destroy (rwlock) == 0)
    result = PASS;

  return result;
}
