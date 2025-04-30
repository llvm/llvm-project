/* Overflow tests for stat, statfs, and lseek functions.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sys/stat.h>
#include <sys/statfs.h>

/* Test for overflows of structures where we ask the kernel to fill them
   in with standard 64-bit syscalls but return them through APIs that
   only expose the low 32 bits of some fields.  */

static inline off_t lseek_overflow (loff_t res)
{
  off_t retval = (off_t) res;
  if (retval == res)
    return retval;

  __set_errno (EOVERFLOW);
  return (off_t) -1;
}

static inline int stat_overflow (struct stat *buf)
{
#if defined __INO_T_MATCHES_INO64_T || !STAT_IS_KERNEL_STAT
  return 0;
#else
  if (buf->__st_ino_pad == 0 && buf->__st_size_pad == 0
      && buf->__st_blocks_pad == 0)
    return 0;

  __set_errno (EOVERFLOW);
  return -1;
#endif
}

/* Note that f_files and f_ffree may validly be a sign-extended -1.  */
static inline int statfs_overflow (struct statfs *buf)
{
#if __STATFS_MATCHES_STATFS64 || !STAT_IS_KERNEL_STAT
  return 0;
#else
  if (buf->__f_blocks_pad == 0 && buf->__f_bfree_pad == 0
      && buf->__f_bavail_pad == 0
      && (buf->__f_files_pad == 0
	  || (buf->f_files == -1U && buf->__f_files_pad == -1))
      && (buf->__f_ffree_pad == 0
	  || (buf->f_ffree == -1U && buf->__f_ffree_pad == -1)))
    return 0;

  __set_errno (EOVERFLOW);
  return -1;
#endif
}
