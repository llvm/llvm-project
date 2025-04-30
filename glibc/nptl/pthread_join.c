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
___pthread_join (pthread_t threadid, void **thread_return)
{
  return __pthread_clockjoin_ex (threadid, thread_return, 0 /* Ignored */,
				 NULL, true);
}
versioned_symbol (libc, ___pthread_join, pthread_join, GLIBC_2_34);
libc_hidden_ver (___pthread_join, __pthread_join)
#ifndef SHARED
strong_alias (___pthread_join, __pthread_join)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libc, ___pthread_join, pthread_join, GLIBC_2_0);
#endif
