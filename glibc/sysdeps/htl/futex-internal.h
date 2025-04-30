/* futex operations for glibc-internal use.  Stub version; do not include
   this file directly.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef STUB_FUTEX_INTERNAL_H
#define STUB_FUTEX_INTERNAL_H

#include <pthread.h>

/* Returns EINVAL if PSHARED is neither PTHREAD_PROCESS_PRIVATE nor
   PTHREAD_PROCESS_SHARED; otherwise, returns 0 if PSHARED is supported, and
   ENOTSUP if not.  */
static __always_inline int
futex_supports_pshared (int pshared)
{
  if (__glibc_likely (pshared == PTHREAD_PROCESS_PRIVATE))
    return 0;
  else if (pshared == PTHREAD_PROCESS_SHARED)
    return 0;
  else
    return EINVAL;
}

#endif  /* futex-internal.h */
