/* shm_unlink -- remove a POSIX shared memory object.  Generic POSIX version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <shlib-compat.h>
#include <shm-directory.h>
#include <string.h>
#include <unistd.h>

/* Remove shared memory object.  */
int
__shm_unlink (const char *name)
{
  struct shmdir_name dirname;
  if (__shm_get_name (&dirname, name, false) != 0)
    {
      __set_errno (ENOENT);
      return -1;
    }

  int result = __unlink (dirname.name);
  if (result < 0 && errno == EPERM)
    __set_errno (EACCES);
  return result;
}
versioned_symbol (libc, __shm_unlink, shm_unlink, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libc, __shm_unlink, shm_unlink, GLIBC_2_2);
#endif
