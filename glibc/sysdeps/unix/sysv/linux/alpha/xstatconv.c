/* Convert between the kernel's `struct stat' format, and libc's.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <kernel_stat.h>
#include <xstatconv.h>
#include <sys/syscall.h>

int
__xstat_conv (int vers, struct kernel_stat *kbuf, void *ubuf)
{
  switch (vers)
    {
    case _STAT_VER_KERNEL:
      *(struct kernel_stat *) ubuf = *kbuf;
      break;

    case _STAT_VER_GLIBC2:
      {
	struct glibc2_stat *buf = ubuf;

	buf->st_dev = kbuf->st_dev;
	buf->st_ino = kbuf->st_ino;
	buf->st_mode = kbuf->st_mode;
	buf->st_nlink = kbuf->st_nlink;
	buf->st_uid = kbuf->st_uid;
	buf->st_gid = kbuf->st_gid;
	buf->st_rdev = kbuf->st_rdev;
	buf->st_size = kbuf->st_size;
	buf->st_atime_sec = kbuf->st_atime_sec;
	buf->st_mtime_sec = kbuf->st_mtime_sec;
	buf->st_ctime_sec = kbuf->st_ctime_sec;
	buf->st_blksize = kbuf->st_blksize;
	buf->st_blocks = kbuf->st_blocks;
	buf->st_flags = kbuf->st_flags;
	buf->st_gen = kbuf->st_gen;
      }
      break;

    case _STAT_VER_GLIBC2_1:
      {
	struct glibc21_stat *buf = ubuf;

	buf->st_dev = kbuf->st_dev;
	buf->st_ino = kbuf->st_ino;
	buf->st_mode = kbuf->st_mode;
	buf->st_nlink = kbuf->st_nlink;
	buf->st_uid = kbuf->st_uid;
	buf->st_gid = kbuf->st_gid;
	buf->st_rdev = kbuf->st_rdev;
	buf->st_size = kbuf->st_size;
	buf->st_atime_sec = kbuf->st_atime_sec;
	buf->st_mtime_sec = kbuf->st_mtime_sec;
	buf->st_ctime_sec = kbuf->st_ctime_sec;
	buf->st_blocks = kbuf->st_blocks;
	buf->st_blksize = kbuf->st_blksize;
	buf->st_flags = kbuf->st_flags;
	buf->st_gen = kbuf->st_gen;
	buf->__pad3 = 0;
	buf->__glibc_reserved[0] = 0;
	buf->__glibc_reserved[1] = 0;
	buf->__glibc_reserved[2] = 0;
	buf->__glibc_reserved[3] = 0;
      }
      break;

    case _STAT_VER_GLIBC2_3_4:
      {
	struct stat64 *buf = ubuf;

	buf->st_dev = kbuf->st_dev;
	buf->st_ino = kbuf->st_ino;
	buf->st_rdev = kbuf->st_rdev;
	buf->st_size = kbuf->st_size;
	buf->st_blocks = kbuf->st_blocks;

	buf->st_mode = kbuf->st_mode;
	buf->st_uid = kbuf->st_uid;
	buf->st_gid = kbuf->st_gid;
	buf->st_blksize = kbuf->st_blksize;
	buf->st_nlink = kbuf->st_nlink;
	buf->__pad0 = 0;

	buf->st_atim.tv_sec = kbuf->st_atime_sec;
	buf->st_atim.tv_nsec = 0;
	buf->st_mtim.tv_sec = kbuf->st_mtime_sec;
	buf->st_mtim.tv_nsec = 0;
	buf->st_ctim.tv_sec = kbuf->st_ctime_sec;
	buf->st_ctim.tv_nsec = 0;

	buf->__glibc_reserved[0] = 0;
	buf->__glibc_reserved[1] = 0;
	buf->__glibc_reserved[2] = 0;
      }
      break;

    default:
      __set_errno (EINVAL);
      return -1;
    }

  return 0;
}
