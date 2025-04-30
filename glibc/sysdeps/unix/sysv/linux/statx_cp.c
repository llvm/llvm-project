/* Struct statx to stat/stat64 conversion for Linux.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <stddef.h>
#include <string.h>
#include <sys/stat.h>

#include <statx_cp.h>

#if !defined(__NR_fstat64) || !defined(__NR_fstatat64)
void
__cp_stat64_statx (struct stat64 *to, struct statx *from)
{
  memset (to, 0, sizeof (struct stat64));
  to->st_dev = ((from->stx_dev_minor & 0xff) | (from->stx_dev_major << 8)
		| ((from->stx_dev_minor & ~0xff) << 12));
  to->st_rdev = ((from->stx_rdev_minor & 0xff) | (from->stx_rdev_major << 8)
		 | ((from->stx_rdev_minor & ~0xff) << 12));
  to->st_ino = from->stx_ino;
  to->st_mode = from->stx_mode;
  to->st_nlink = from->stx_nlink;
  to->st_uid = from->stx_uid;
  to->st_gid = from->stx_gid;
  to->st_atime = from->stx_atime.tv_sec;
  to->st_atim.tv_nsec = from->stx_atime.tv_nsec;
  to->st_mtime = from->stx_mtime.tv_sec;
  to->st_mtim.tv_nsec = from->stx_mtime.tv_nsec;
  to->st_ctime = from->stx_ctime.tv_sec;
  to->st_ctim.tv_nsec = from->stx_ctime.tv_nsec;
  to->st_size = from->stx_size;
  to->st_blocks = from->stx_blocks;
  to->st_blksize = from->stx_blksize;
}
#endif

