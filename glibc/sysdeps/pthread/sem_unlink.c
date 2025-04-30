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
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "semaphoreP.h"
#include <shm-directory.h>

#if !PTHREAD_IN_LIBC
/* The private name is not exported from libc.  */
# define __unlink unlink
#endif

int
__sem_unlink (const char *name)
{
  struct shmdir_name dirname;
  if (__shm_get_name (&dirname, name, true) != 0)
    {
      __set_errno (ENOENT);
      return -1;
    }

  /* Now try removing it.  */
  int ret = __unlink (dirname.name);
  if (ret < 0 && errno == EPERM)
    __set_errno (EACCES);
  return ret;
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __sem_unlink, sem_unlink, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1_1, GLIBC_2_34)
compat_symbol (libpthread, __sem_unlink, sem_unlink, GLIBC_2_1_1);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__sem_unlink, sem_unlink)
#endif
