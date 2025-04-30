/* C11 threads call once implementation.
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

#include <stdalign.h>
#include <shlib-compat.h>

#include "thrd_priv.h"

void
__call_once (once_flag *flag, void (*func)(void))
{
  _Static_assert (sizeof (once_flag) == sizeof (pthread_once_t),
		  "sizeof (once_flag) != sizeof (pthread_once_t)");
  _Static_assert (alignof (once_flag) == alignof (pthread_once_t),
		  "alignof (once_flag) != alignof (pthread_once_t)");
  __pthread_once ((pthread_once_t *) flag, func);
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __call_once, call_once, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_28, GLIBC_2_34)
compat_symbol (libpthread, __call_once, call_once, GLIBC_2_28);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__call_once, call_once)
#endif
