/* Return information about the filesystem on which FD resides.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#define __fstatvfs __fstatvfs_disable
#define fstatvfs fstatvfs_disable
#include <sys/statvfs.h>
#include <sys/statfs.h>
#include <internal_statvfs.h>
#include <time.h>
#include <kernel_stat.h>

/* Return information about the filesystem on which FD resides.  */
int
__fstatvfs64 (int fd, struct statvfs64 *buf)
{
  struct statfs64 fsbuf;
  if (__fstatfs64 (fd, &fsbuf) < 0)
    return -1;

  /* Convert the result.  */
  __internal_statvfs64 (buf, &fsbuf);

  return 0;
}
weak_alias (__fstatvfs64, fstatvfs64)

#undef __fstatvfs
#undef fstatvfs

#if STATFS_IS_STATFS64
weak_alias (__fstatvfs64, __fstatvfs)
weak_alias (__fstatvfs64, fstatvfs)
#endif
