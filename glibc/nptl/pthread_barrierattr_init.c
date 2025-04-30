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
#include <sys/mman.h>

int
__pthread_barrierattr_init (pthread_barrierattr_t *attr)
{
  ASSERT_TYPE_SIZE (pthread_barrierattr_t, __SIZEOF_PTHREAD_BARRIERATTR_T);
  ASSERT_PTHREAD_INTERNAL_SIZE (pthread_barrierattr_t,
				struct pthread_barrierattr);

  ((struct pthread_barrierattr *) attr)->pshared = PTHREAD_PROCESS_PRIVATE;

  __try_to_mark_as_unmigratable (attr);

  return 0;
}
versioned_symbol (libc, __pthread_barrierattr_init,
                  pthread_barrierattr_init, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_barrierattr_init,
               pthread_barrierattr_init, GLIBC_2_2);
#endif
