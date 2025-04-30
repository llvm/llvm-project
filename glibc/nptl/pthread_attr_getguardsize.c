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

int
__pthread_attr_getguardsize (const pthread_attr_t *attr, size_t *guardsize)
{
  struct pthread_attr *iattr;

  iattr = (struct pthread_attr *) attr;

  *guardsize = iattr->guardsize;

  return 0;
}
versioned_symbol (libpthread, __pthread_attr_getguardsize,
                  pthread_attr_getguardsize, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __pthread_attr_getguardsize,
               pthread_attr_getguardsize, GLIBC_2_1);
#endif
