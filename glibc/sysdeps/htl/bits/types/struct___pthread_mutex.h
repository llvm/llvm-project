/* Mutex type.  Generic version.

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

#ifndef _BITS_TYPES_STRUCT___PTHREAD_MUTEX_H
#define _BITS_TYPES_STRUCT___PTHREAD_MUTEX_H	1

#include <bits/types/__pthread_spinlock_t.h>
#include <bits/types/struct___pthread_mutexattr.h>

/* User visible part of a mutex.  */
struct __pthread_mutex
{
  __pthread_spinlock_t __held;
  __pthread_spinlock_t __lock;
  /* In cthreads, mutex_init does not initialized thre third
     pointer, as such, we cannot rely on its value for anything.  */
  char *__cthreadscompat1;
  struct __pthread *__queue;
  struct __pthread_mutexattr *__attr;
  void *__data;
  /*  Up to this point, we are completely compatible with cthreads
     and what libc expects.  */
  void *__owner;
  unsigned __locks;
  /* If NULL then the default attributes apply.  */
};

/* Initializer for a mutex.  N.B.  this also happens to be compatible
   with the cthread mutex initializer.  */
#define __PTHREAD_MUTEX_INITIALIZER \
    { __PTHREAD_SPIN_LOCK_INITIALIZER, __PTHREAD_SPIN_LOCK_INITIALIZER, 0, 0, 0, 0, 0, 0 }

#define __PTHREAD_ERRORCHECK_MUTEXATTR ((struct __pthread_mutexattr *) ((unsigned long) __PTHREAD_MUTEX_ERRORCHECK + 1))

#define __PTHREAD_ERRORCHECK_MUTEX_INITIALIZER \
    { __PTHREAD_SPIN_LOCK_INITIALIZER, __PTHREAD_SPIN_LOCK_INITIALIZER, 0, 0,	\
	__PTHREAD_ERRORCHECK_MUTEXATTR, 0, 0, 0 }

#define __PTHREAD_RECURSIVE_MUTEXATTR ((struct __pthread_mutexattr *) ((unsigned long) __PTHREAD_MUTEX_RECURSIVE + 1))

#define __PTHREAD_RECURSIVE_MUTEX_INITIALIZER \
    { __PTHREAD_SPIN_LOCK_INITIALIZER, __PTHREAD_SPIN_LOCK_INITIALIZER, 0, 0,	\
	__PTHREAD_RECURSIVE_MUTEXATTR, 0, 0, 0 }

#endif /* bits/types/struct___pthread_mutex.h */
