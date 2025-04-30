/* Internal definitions for pthreads library.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _PT_MUTEX_H
#define _PT_MUTEX_H	1

/* Special ID used to signal an unrecoverable robust mutex. */
#define NOTRECOVERABLE_ID   (1U << 31)

/* Common path for robust mutexes. Assumes the variable 'ret'
 * is bound in the function this is called from. */
#define ROBUST_LOCK(self, mtxp, cb, ...)   \
  if (mtxp->__owner_id == NOTRECOVERABLE_ID)   \
    return ENOTRECOVERABLE;   \
  else if (mtxp->__owner_id == self->thread   \
	   && __getpid () == (int)(mtxp->__lock & LLL_OWNER_MASK))   \
    {   \
      if (mtxp->__type == PT_MTX_RECURSIVE)   \
        {   \
          if (__glibc_unlikely (mtxp->__cnt + 1 == 0))   \
            return EAGAIN;   \
          \
          ++mtxp->__cnt;   \
          return 0;   \
        }   \
      else if (mtxp->__type == PT_MTX_ERRORCHECK)   \
        return EDEADLK;   \
    }   \
  \
  ret = cb (mtxp->__lock, ##__VA_ARGS__);   \
  if (ret == 0 || ret == EOWNERDEAD)   \
    {   \
      if (mtxp->__owner_id == ENOTRECOVERABLE)   \
        ret = ENOTRECOVERABLE;   \
      else   \
        {   \
          mtxp->__owner_id = self->thread;   \
          mtxp->__cnt = 1;   \
          if (ret == EOWNERDEAD)   \
            {   \
              mtxp->__lock = mtxp->__lock | LLL_DEAD_OWNER;   \
              atomic_write_barrier ();   \
            }   \
        }   \
    }   \
  (void)0

/* Check that a thread owns the mutex. For non-robust, task-shared
 * objects, we have to check the thread *and* process-id. */
#define mtx_owned_p(mtx, pt, flags)   \
  ((mtx)->__owner_id == (pt)->thread   \
   && (((flags) & GSYNC_SHARED) == 0   \
       || (mtx)->__shpid == __getpid ()))

/* Record a thread as the owner of the mutex. */
#define mtx_set_owner(mtx, pt, flags)   \
  (void)   \
    ({   \
       (mtx)->__owner_id = (pt)->thread;   \
       if ((flags) & GSYNC_SHARED)   \
         (mtx)->__shpid = __getpid ();   \
     })

/* Redefined mutex types. The +1 is for binary compatibility. */
#define PT_MTX_NORMAL       __PTHREAD_MUTEX_TIMED
#define PT_MTX_RECURSIVE    (__PTHREAD_MUTEX_RECURSIVE + 1)
#define PT_MTX_ERRORCHECK   (__PTHREAD_MUTEX_ERRORCHECK + 1)

/* Mutex type, including robustness. */
#define MTX_TYPE(mtxp)   \
  ((mtxp)->__type | ((mtxp)->__flags & PTHREAD_MUTEX_ROBUST))

extern int __getpid (void) __attribute__ ((const));

#endif /* pt-mutex.h */
