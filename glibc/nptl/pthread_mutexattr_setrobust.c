/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2005.

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
#include <pthreadP.h>
#include <shlib-compat.h>

int
__pthread_mutexattr_setrobust (pthread_mutexattr_t *attr, int robustness)
{
  if (robustness != PTHREAD_MUTEX_STALLED_NP
      && __builtin_expect (robustness != PTHREAD_MUTEX_ROBUST_NP, 0))
    return EINVAL;

  struct pthread_mutexattr *iattr = (struct pthread_mutexattr *) attr;

  /* We use bit 30 to signal whether the mutex is going to be
     robust or not.  */
  if (robustness == PTHREAD_MUTEX_STALLED_NP)
    iattr->mutexkind &= ~PTHREAD_MUTEXATTR_FLAG_ROBUST;
  else
    iattr->mutexkind |= PTHREAD_MUTEXATTR_FLAG_ROBUST;

  return 0;
}
versioned_symbol (libc, __pthread_mutexattr_setrobust,
                  pthread_mutexattr_setrobust, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_4, GLIBC_2_34)
compat_symbol (libpthread, __pthread_mutexattr_setrobust,
               pthread_mutexattr_setrobust_np, GLIBC_2_4);
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_12, GLIBC_2_34)
compat_symbol (libpthread, __pthread_mutexattr_setrobust,
               pthread_mutexattr_setrobust, GLIBC_2_12);
#endif
