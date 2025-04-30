/* rwlock type.  Generic version.
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

#ifndef _BITS_TYPES_STRUCT___PTHREAD_RWLOCK_H
#define _BITS_TYPES_STRUCT___PTHREAD_RWLOCK_H

#include <bits/types/__pthread_spinlock_t.h>

/* User visible part of a rwlock.  If __held is not held and readers
   is 0, then the lock is unlocked.  If __held is held and readers is
   0, then the lock is held by a writer.  If __held is held and
   readers is greater than 0, then the lock is held by READERS
   readers.  */
struct __pthread_rwlock
{
  __pthread_spinlock_t __held;
  __pthread_spinlock_t __lock;
  int __readers;
  struct __pthread *__readerqueue;
  struct __pthread *__writerqueue;
  struct __pthread_rwlockattr *__attr;
  void *__data;
};

/* Initializer for a rwlock.  */
#define __PTHREAD_RWLOCK_INITIALIZER \
    { __PTHREAD_SPIN_LOCK_INITIALIZER, __PTHREAD_SPIN_LOCK_INITIALIZER, 0, 0, 0, 0, 0 }


#endif /* bits/types/struct___pthread_rwlock.h */
