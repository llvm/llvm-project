/* C11 threads thread exit implementation.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>
#include "thrd_priv.h"

_Noreturn void
__thrd_exit (int res)
{
  __pthread_exit ((void*)(uintptr_t) res);
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __thrd_exit, thrd_exit, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_28, GLIBC_2_34)
compat_symbol (libpthread, __thrd_exit, thrd_exit, GLIBC_2_28);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__thrd_exit, thrd_exit)
#endif
