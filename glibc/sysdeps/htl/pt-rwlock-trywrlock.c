/* Try to acquire a rwlock for writing.  Generic version.
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
#include <assert.h>

#include <pt-internal.h>

/* Try to acquire RWLOCK for writing.  */
int
pthread_rwlock_trywrlock (struct __pthread_rwlock *rwlock)
{
  __pthread_spin_wait (&rwlock->__lock);
  if (__pthread_spin_trylock (&rwlock->__held) == 0)
    /* Successfully acquired the lock.  */
    {
      assert (rwlock->__readerqueue == 0);
      assert (rwlock->__writerqueue == 0);
      assert (rwlock->__readers == 0);

      __pthread_spin_unlock (&rwlock->__lock);
      return 0;
    }

  /* The lock is busy.  */

  __pthread_spin_unlock (&rwlock->__lock);

  return EBUSY;
}
