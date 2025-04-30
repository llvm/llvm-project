/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <stddef.h>
#include <sys/stat.h>
#include <hurd.h>
#include <shlib-compat.h>

#if LIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_33)

/* Get information about the file descriptor FD in BUF.  */
int
__fxstat (int vers, int fd, struct stat *buf)
{
  if (vers != _STAT_VER)
    return __hurd_fail (EINVAL);

  return __fstat (fd, buf);
}
weak_alias (__fxstat, _fxstat)

#endif /* LIB_COMPAT  */
