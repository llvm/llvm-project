/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include "pthreadP.h"
#include <string.h>
#include <pthread-offsets.h>
#include <shlib-compat.h>
#include <sys/mman.h>

static const struct pthread_rwlockattr default_rwlockattr =
  {
    .lockkind = PTHREAD_RWLOCK_DEFAULT_NP,
    .pshared = PTHREAD_PROCESS_PRIVATE
  };


/* See pthread_rwlock_common.c.  */
int
___pthread_rwlock_init (pthread_rwlock_t *rwlock,
		       const pthread_rwlockattr_t *attr)
{
  ASSERT_TYPE_SIZE (pthread_rwlock_t, __SIZEOF_PTHREAD_RWLOCK_T);

  /* The __flags is the only field where its offset should be checked to
     avoid ABI breakage with static initializers.  */
  ASSERT_PTHREAD_INTERNAL_OFFSET (pthread_rwlock_t, __data.__flags,
				  __PTHREAD_RWLOCK_FLAGS_OFFSET);
  ASSERT_PTHREAD_INTERNAL_MEMBER_SIZE (pthread_rwlock_t, __data.__flags,
				       int);

  const struct pthread_rwlockattr *iattr;

  iattr = ((const struct pthread_rwlockattr *) attr) ?: &default_rwlockattr;

  memset (rwlock, '\0', sizeof (*rwlock));

  rwlock->__data.__flags = iattr->lockkind;

  /* The value of __SHARED in a private rwlock must be zero.  */
  rwlock->__data.__shared = (iattr->pshared != PTHREAD_PROCESS_PRIVATE);

  __try_to_mark_as_unmigratable (rwlock);

  return 0;
}
versioned_symbol (libc, ___pthread_rwlock_init, pthread_rwlock_init,
                  GLIBC_2_34);
libc_hidden_ver (___pthread_rwlock_init, __pthread_rwlock_init)
#ifndef SHARED
strong_alias (___pthread_rwlock_init, __pthread_rwlock_init)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_rwlock_init, pthread_rwlock_init,
               GLIBC_2_1);
#endif
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_rwlock_init, __pthread_rwlock_init,
               GLIBC_2_2);
#endif
