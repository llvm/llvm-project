/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <semaphore.h>
#include <futex-internal.h>
#include "pthreadP.h"

#define SEM_SHM_PREFIX  "sem."

static inline void __new_sem_open_init (struct new_sem *sem, unsigned value)
{
#if __HAVE_64B_ATOMICS
  sem->data = value;
#else
  sem->value = value << SEM_VALUE_SHIFT;
  sem->nwaiters = 0;
#endif
  /* pad is used as a mutex on pre-v9 sparc and ignored otherwise.  */
  sem->pad = 0;

  /* This always is a shared semaphore.  */
  sem->private = FUTEX_SHARED;
}

/* Prototypes of functions with multiple interfaces.  */
extern int __new_sem_init (sem_t *sem, int pshared, unsigned int value);
extern int __old_sem_init (sem_t *sem, int pshared, unsigned int value);
extern int __new_sem_destroy (sem_t *sem);
extern int __new_sem_post (sem_t *sem);
extern int __new_sem_wait (sem_t *sem);
extern int __old_sem_wait (sem_t *sem);
extern int __new_sem_trywait (sem_t *sem);
extern int __new_sem_getvalue (sem_t *sem, int *sval);

#if __TIMESIZE == 64
# define __sem_clockwait64 __sem_clockwait
# define __sem_timedwait64 __sem_timedwait
#else
extern int
__sem_clockwait64 (sem_t *sem, clockid_t clockid,
                   const struct __timespec64 *abstime);
libc_hidden_proto (__sem_clockwait64)
extern int
__sem_timedwait64 (sem_t *sem, const struct __timespec64 *abstime);
libc_hidden_proto (__sem_timedwait64)
#endif
