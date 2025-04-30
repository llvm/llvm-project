/* pthread_key internal declatations for the Hurd version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <libc-lockP.h>
#include <pthreadP.h>

#define PTHREAD_KEY_MEMBERS \
  void **thread_specifics;		/* This is only resized by the thread, and always growing */ \
  unsigned thread_specifics_size;	/* Number of entries in thread_specifics */

#define PTHREAD_KEY_INVALID (void *) (-1)


/* __PTHREAD_KEY_DESTRUCTORS is an array of destructors with
   __PTHREAD_KEY_SIZE elements.  If an element with index less than
   __PTHREAD_KEY_COUNT is invalid, it shall contain the value
   PTHREAD_KEY_INVALID which shall be distinct from NULL.

   Normally, we just add new keys to the end of the array and realloc
   it as necessary.  The pthread_key_create routine may decide to
   rescan the array if __PTHREAD_KEY_FREE is large.  */
extern void (**__pthread_key_destructors) (void *arg);
extern int __pthread_key_size;
extern int __pthread_key_count;
/* Number of invalid elements in the array.  Does not include elements
   for which memory has been allocated but which have not yet been
   used (i.e. those elements with indexes greater than
   __PTHREAD_KEY_COUNT).  */
extern int __pthread_key_invalid_count;

/* Protects the above variables.  This must be a recursive lock: the
   destructors may call pthread_key_delete.  */
extern pthread_mutex_t __pthread_key_lock;

#include <assert.h>

static inline void
__pthread_key_lock_ready (void)
{
  static pthread_once_t o = PTHREAD_ONCE_INIT;

  void do_init (void)
  {
    int err;
    pthread_mutexattr_t attr;

    err = __pthread_mutexattr_init (&attr);
    assert_perror (err);

    err = __pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_RECURSIVE);
    assert_perror (err);

    err = __pthread_mutex_init (&__pthread_key_lock, &attr);
    assert_perror (err);

    err = __pthread_mutexattr_destroy (&attr);
    assert_perror (err);
  }

  __pthread_once (&o, do_init);
}
