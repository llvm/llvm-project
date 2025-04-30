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

/* Global definition.  Needed in pthread_getconcurrency as well.  */
int __concurrency_level;


int
__pthread_setconcurrency (int level)
{
  if (level < 0)
    return EINVAL;

  __concurrency_level = level;

  /* XXX For ports which actually need to handle the concurrency level
     some more code is probably needed here.  */

  return 0;
}
versioned_symbol (libc, __pthread_setconcurrency, pthread_setconcurrency,
                  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __pthread_setconcurrency, pthread_setconcurrency,
               GLIBC_2_1);
#endif
