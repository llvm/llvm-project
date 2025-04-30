/* Get filesystem statistics.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <sys/statfs.h>
#include <time.h>
#include <sysdep.h>
#include <kernel_stat.h>

#if !STATFS_IS_STATFS64

/* Return information about the filesystem on which FILE resides.  */
int
__fstatfs (int fd, struct statfs *buf)
{
  struct statfs64 buf64;
  int rc = INLINE_SYSCALL_CALL (fstatfs64, fd, sizeof (buf64), &buf64);
  if (rc != 0)
    return rc;

  buf->f_type = buf64.f_type;
  buf->f_bsize = buf64.f_bsize;
  buf->f_blocks = buf64.f_blocks;
  buf->f_bfree = buf64.f_bfree;
  buf->f_bavail = buf64.f_bavail;
  buf->f_files = buf64.f_files;
  buf->f_ffree = buf64.f_ffree;
  buf->f_fsid = buf64.f_fsid;
  buf->f_namelen = buf64.f_namelen;
  buf->f_frsize = buf64.f_frsize;
  buf->f_flags = buf64.f_flags;
  memcpy (buf->f_spare, buf64.f_spare, sizeof (buf64.f_spare));

  if ((fsblkcnt_t) buf64.f_blocks != buf64.f_blocks
      || (fsblkcnt_t) buf64.f_bfree != buf64.f_bfree
      || (fsblkcnt_t) buf64.f_bavail != buf64.f_bavail
      || (fsblkcnt_t) buf64.f_files != buf64.f_files
      || (fsblkcnt_t) buf64.f_ffree != buf64.f_ffree)
   {
     __set_errno (EOVERFLOW);
     return -1;
   }

  return 0;
}
libc_hidden_def (__fstatfs)
weak_alias (__fstatfs, fstatfs)
#endif
