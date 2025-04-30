/* Get filesystem statistics.  Linux/alpha.
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

#include <sys/statfs.h>
#include <sysdep.h>
#include <kernel_stat.h>

/* Return information about the filesystem on which FILE resides.  */
int
__fstatfs64 (int fd, struct statfs64 *buf)
{
  int r = INLINE_SYSCALL_CALL (fstatfs64, fd, sizeof (*buf), buf);
#if __ASSUME_STATFS64 == 0
  if (r == -1 && errno == ENOSYS)
    {
      struct statfs buf32;
      if (__fstatfs (fd, &buf32) < 0)
	return -1;

      buf->f_type = buf32.f_type;
      buf->f_bsize = buf32.f_bsize;
      buf->f_blocks = buf32.f_blocks;
      buf->f_bfree = buf32.f_bfree;
      buf->f_bavail = buf32.f_bavail;
      buf->f_files = buf32.f_files;
      buf->f_ffree = buf32.f_ffree;
      buf->f_fsid = buf32.f_fsid;
      buf->f_namelen = buf32.f_namelen;
      buf->f_frsize = buf32.f_frsize;
      buf->f_flags = buf32.f_flags;
      memcpy (buf->f_spare, buf32.f_spare, sizeof (buf32.f_spare));
    }
#endif
  return r;
}
weak_alias (__fstatfs64, fstatfs64)
