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
#include <shlib-compat.h>
#include <sys/mman.h>

int
__pthread_rwlockattr_init (pthread_rwlockattr_t *attr)
{
  ASSERT_TYPE_SIZE (pthread_rwlockattr_t, __SIZEOF_PTHREAD_RWLOCKATTR_T);
  ASSERT_PTHREAD_INTERNAL_SIZE (pthread_rwlockattr_t,
				struct pthread_rwlockattr);

  struct pthread_rwlockattr *iattr;

  iattr = (struct pthread_rwlockattr *) attr;

  iattr->lockkind = PTHREAD_RWLOCK_DEFAULT_NP;
  iattr->pshared = PTHREAD_PROCESS_PRIVATE;

  __try_to_mark_as_unmigratable (iattr);

  return 0;
}
versioned_symbol (libc, __pthread_rwlockattr_init,
                  pthread_rwlockattr_init, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __pthread_rwlockattr_init,
               pthread_rwlockattr_init, GLIBC_2_1);
#endif
