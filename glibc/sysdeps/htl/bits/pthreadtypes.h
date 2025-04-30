/* Declaration of common pthread types for all architectures.  Hurd version.
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

#if !defined _BITS_TYPES_H && !defined _PTHREAD_H
# error "Never include <bits/pthreadtypes.h> directly; use <sys/types.h> instead."
#endif

#ifndef _BITS_PTHREADTYPES_H
#define _BITS_PTHREADTYPES_H    1

#include <bits/thread-shared-types.h>

#include <features.h>

#include <bits/types.h>

__BEGIN_DECLS
#include <bits/pthread.h>
typedef __pthread_t pthread_t;

/* Possible values for the process shared attribute.  */
enum __pthread_process_shared
{
  __PTHREAD_PROCESS_PRIVATE = 0,
  __PTHREAD_PROCESS_SHARED
};

/* Possible values for the inheritsched attribute.  */
enum __pthread_inheritsched
{
  __PTHREAD_EXPLICIT_SCHED = 0,
  __PTHREAD_INHERIT_SCHED
};

/* Possible values for the `contentionscope' attribute.  */
enum __pthread_contentionscope
{
  __PTHREAD_SCOPE_SYSTEM = 0,
  __PTHREAD_SCOPE_PROCESS
};

/* Possible values for the `detachstate' attribute.  */
enum __pthread_detachstate
{
  __PTHREAD_CREATE_JOINABLE = 0,
  __PTHREAD_CREATE_DETACHED
};

#include <bits/types/struct___pthread_attr.h>
typedef struct __pthread_attr pthread_attr_t;

enum __pthread_mutex_protocol
{
  __PTHREAD_PRIO_NONE = 0,
  __PTHREAD_PRIO_INHERIT,
  __PTHREAD_PRIO_PROTECT
};

enum __pthread_mutex_type
{
  __PTHREAD_MUTEX_TIMED,
  __PTHREAD_MUTEX_ERRORCHECK,
  __PTHREAD_MUTEX_RECURSIVE
};

enum __pthread_mutex_robustness
{
  __PTHREAD_MUTEX_STALLED,
  __PTHREAD_MUTEX_ROBUST = 0x100
};

#include <bits/types/struct___pthread_mutexattr.h>
typedef struct __pthread_mutexattr pthread_mutexattr_t;

#include <bits/types/struct___pthread_mutex.h>
typedef struct __pthread_mutex pthread_mutex_t;

#include <bits/types/struct___pthread_condattr.h>
typedef struct __pthread_condattr pthread_condattr_t;

#include <bits/types/struct___pthread_cond.h>
typedef struct __pthread_cond pthread_cond_t;

#ifdef __USE_XOPEN2K
# include <bits/types/__pthread_spinlock_t.h>
typedef __pthread_spinlock_t pthread_spinlock_t;
#endif /* XPG6.  */

#if defined __USE_UNIX98 || defined __USE_XOPEN2K

# include <bits/types/struct___pthread_rwlockattr.h>
typedef struct __pthread_rwlockattr pthread_rwlockattr_t;

# include <bits/types/struct___pthread_rwlock.h>
typedef struct __pthread_rwlock pthread_rwlock_t;

#endif /* __USE_UNIX98 || __USE_XOPEN2K */

#ifdef __USE_XOPEN2K

# include <bits/types/struct___pthread_barrierattr.h>
typedef struct __pthread_barrierattr pthread_barrierattr_t;

# include <bits/types/struct___pthread_barrier.h>
typedef struct __pthread_barrier pthread_barrier_t;

#endif /* __USE_XOPEN2K */

#include <bits/types/__pthread_key.h>
typedef __pthread_key pthread_key_t;

#include <bits/types/struct___pthread_once.h>
typedef struct __pthread_once pthread_once_t;

__END_DECLS
#endif /* bits/pthreadtypes.h */
