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

#include <shlib-compat.h>

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
/* This function does not serve a useful purpose in the thread library
   implementation anymore.  It used to be necessary when then kernel
   could not shut down "processes" but this is not the case anymore.

   We could theoretically provide an equivalent implementation but
   this is not necessary since the kernel already does a much better
   job than we ever could.  */
void
__pthread_kill_other_threads_np (void)
{
}
compat_symbol (libpthread, __pthread_kill_other_threads_np,
	       pthread_kill_other_threads_np, GLIBC_2_0);
#endif
