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

#include <errno.h>
#include "pthreadP.h"
#include <shlib-compat.h>

int
__pthread_rwlockattr_setkind_np (pthread_rwlockattr_t *attr, int pref)
{
  struct pthread_rwlockattr *iattr;

  if (pref != PTHREAD_RWLOCK_PREFER_READER_NP
      && pref != PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP
      && __builtin_expect  (pref != PTHREAD_RWLOCK_PREFER_WRITER_NP, 0))
    return EINVAL;

  iattr = (struct pthread_rwlockattr *) attr;

  iattr->lockkind = pref;

  return 0;
}
versioned_symbol (libc, __pthread_rwlockattr_setkind_np,
                  pthread_rwlockattr_setkind_np, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __pthread_rwlockattr_setkind_np,
               pthread_rwlockattr_setkind_np, GLIBC_2_1);
#endif
