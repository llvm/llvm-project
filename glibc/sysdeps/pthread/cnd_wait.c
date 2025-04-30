/* C11 threads conditional wait implementaiton.
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

#include "thrd_priv.h"
#include <shlib-compat.h>

int
__cnd_wait (cnd_t *cond, mtx_t *mutex)
{
  int err_code = __pthread_cond_wait ((pthread_cond_t *) cond,
				      (pthread_mutex_t *) mutex);
  return thrd_err_map (err_code);
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __cnd_wait, cnd_wait, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_28, GLIBC_2_34)
compat_symbol (libpthread, __cnd_wait, cnd_wait, GLIBC_2_28);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__cnd_wait, cnd_wait)
#endif
