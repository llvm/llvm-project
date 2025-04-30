/* Get the default attributes used by pthread_create in the process.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthreadP.h>
#include <shlib-compat.h>

int
___pthread_getattr_default_np (pthread_attr_t *out)
{
  lll_lock (__default_pthread_attr_lock, LLL_PRIVATE);
  int ret = __pthread_attr_copy (out, &__default_pthread_attr.external);
  lll_unlock (__default_pthread_attr_lock, LLL_PRIVATE);
  return ret;
}
versioned_symbol (libc, ___pthread_getattr_default_np,
                  pthread_getattr_default_np, GLIBC_2_34);
libc_hidden_ver (___pthread_getattr_default_np, __pthread_getattr_default_np)
#ifndef SHARED
strong_alias (___pthread_getattr_default_np, __pthread_getattr_default_np)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_18, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_getattr_default_np,
               pthread_getattr_default_np, GLIBC_2_18);
#endif
