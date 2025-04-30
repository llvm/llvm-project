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
#include <pthreadP.h>
#include <shlib-compat.h>

int
___pthread_mutexattr_settype (pthread_mutexattr_t *attr, int kind)
{
  struct pthread_mutexattr *iattr;

  if (kind < PTHREAD_MUTEX_NORMAL || kind > PTHREAD_MUTEX_ADAPTIVE_NP)
    return EINVAL;

  /* Cannot distinguish between DEFAULT and NORMAL. So any settype
     call disables elision for now.  */
  if (kind == PTHREAD_MUTEX_NORMAL)
    kind |= PTHREAD_MUTEX_NO_ELISION_NP;

  iattr = (struct pthread_mutexattr *) attr;

  iattr->mutexkind = (iattr->mutexkind & PTHREAD_MUTEXATTR_FLAG_BITS) | kind;

  return 0;
}
versioned_symbol (libc, ___pthread_mutexattr_settype,
                  pthread_mutexattr_settype, GLIBC_2_34);
libc_hidden_ver (___pthread_mutexattr_settype, __pthread_mutexattr_settype)
#ifndef SHARED
strong_alias (___pthread_mutexattr_settype, __pthread_mutexattr_settype)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutexattr_settype,
               pthread_mutexattr_setkind_np, GLIBC_2_0);
compat_symbol (libpthread, ___pthread_mutexattr_settype,
               __pthread_mutexattr_settype, GLIBC_2_0);
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutexattr_settype,
               pthread_mutexattr_settype, GLIBC_2_1);
#endif
