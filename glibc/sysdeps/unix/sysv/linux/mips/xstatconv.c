/* Convert between the kernel's `struct stat' format, and libc's.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/stat.h>
#include <kernel_stat.h>

#include <string.h>

int
__xstat_conv (int vers, struct kernel_stat *kbuf, void *ubuf)
{
  switch (vers)
    {
    case _STAT_VER_KERNEL:
      /* Nothing to do.  The struct is in the form the kernel expects.
         We should have short-circuted before we got here, but for
         completeness... */
      *(struct kernel_stat *) ubuf = *kbuf;
      break;

    case _STAT_VER_LINUX:
      {
	struct stat *buf = ubuf;

	/* Convert to current kernel version of `struct stat'.  */
	buf->st_dev = kbuf->st_dev;
	memset (&buf->st_pad1, 0, sizeof (buf->st_pad1));
	buf->st_ino = kbuf->st_ino;
	/* Check for overflow.  */
	if (buf->st_ino != kbuf->st_ino)
	  {
	    __set_errno (EOVERFLOW);
	    return -1;
	  }
	buf->st_mode = kbuf->st_mode;
	buf->st_nlink = kbuf->st_nlink;
	buf->st_uid = kbuf->st_uid;
	buf->st_gid = kbuf->st_gid;
	buf->st_rdev = kbuf->st_rdev;
	memset (&buf->st_pad2, 0, sizeof (buf->st_pad2));
	buf->st_size = kbuf->st_size;
	/* Check for overflow.  */
	if (buf->st_size != kbuf->st_size)
	  {
	    __set_errno (EOVERFLOW);
	    return -1;
	  }
	buf->st_pad3 = 0;
	buf->st_atim.tv_sec = kbuf->st_atime_sec;
	buf->st_atim.tv_nsec = kbuf->st_atime_nsec;
	buf->st_mtim.tv_sec = kbuf->st_mtime_sec;
	buf->st_mtim.tv_nsec = kbuf->st_mtime_nsec;
	buf->st_ctim.tv_sec = kbuf->st_ctime_sec;
	buf->st_ctim.tv_nsec = kbuf->st_ctime_nsec;
	buf->st_blksize = kbuf->st_blksize;
	buf->st_blocks = kbuf->st_blocks;
	/* Check for overflow.  */
	if (buf->st_blocks != kbuf->st_blocks)
	  {
	    __set_errno (EOVERFLOW);
	    return -1;
	  }
	memset (&buf->st_pad5, 0, sizeof (buf->st_pad5));
      }
      break;

    default:
      __set_errno (EINVAL);
      return -1;
    }

  return 0;
}

int
__xstat64_conv (int vers, struct kernel_stat *kbuf, void *ubuf)
{
#if XSTAT_IS_XSTAT64
  return xstat_conv (vers, kbuf, ubuf);
#else
  switch (vers)
    {
    case _STAT_VER_LINUX:
      {
	struct stat64 *buf = ubuf;

	buf->st_dev = kbuf->st_dev;
	memset (&buf->st_pad1, 0, sizeof (buf->st_pad1));
	buf->st_ino = kbuf->st_ino;
	buf->st_mode = kbuf->st_mode;
	buf->st_nlink = kbuf->st_nlink;
	buf->st_uid = kbuf->st_uid;
	buf->st_gid = kbuf->st_gid;
	buf->st_rdev = kbuf->st_rdev;
	memset (&buf->st_pad2, 0, sizeof (buf->st_pad2));
	buf->st_pad3 = 0;
	buf->st_size = kbuf->st_size;
	buf->st_blksize = kbuf->st_blksize;
	buf->st_blocks = kbuf->st_blocks;

	buf->st_atim.tv_sec = kbuf->st_atime_sec;
	buf->st_atim.tv_nsec = kbuf->st_atime_nsec;
	buf->st_mtim.tv_sec = kbuf->st_mtime_sec;
	buf->st_mtim.tv_nsec = kbuf->st_mtime_nsec;
	buf->st_ctim.tv_sec = kbuf->st_ctime_sec;
	buf->st_ctim.tv_nsec = kbuf->st_ctime_nsec;

	memset (&buf->st_pad4, 0, sizeof (buf->st_pad4));
      }
      break;

      /* If struct stat64 is different from struct stat then
	 _STAT_VER_KERNEL does not make sense.  */
    case _STAT_VER_KERNEL:
    default:
      __set_errno (EINVAL);
      return -1;
    }

  return 0;
#endif
}

#if _MIPS_SIM == _ABIO32
int
__xstat32_conv (int vers, struct stat64 *kbuf, struct stat *buf)
{
  switch (vers)
    {
    case _STAT_VER_LINUX:
      /* Convert current kernel version of `struct stat64' to
	 `struct stat'.  The layout of the fields in the kernel's
	 stat64 is the same as that in the user stat64; the only
	 difference is that the latter has more trailing padding.  */
      buf->st_dev = kbuf->st_dev;
      memset (&buf->st_pad1, 0, sizeof (buf->st_pad1));
      buf->st_ino = kbuf->st_ino;
      /* Check for overflow.  */
      if (buf->st_ino != kbuf->st_ino)
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}
      buf->st_mode = kbuf->st_mode;
      buf->st_nlink = kbuf->st_nlink;
      buf->st_uid = kbuf->st_uid;
      buf->st_gid = kbuf->st_gid;
      buf->st_rdev = kbuf->st_rdev;
      memset (&buf->st_pad2, 0, sizeof (buf->st_pad2));
      buf->st_size = kbuf->st_size;
      /* Check for overflow.  */
      if (buf->st_size != kbuf->st_size)
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}
      buf->st_pad3 = 0;
      buf->st_atim.tv_sec = kbuf->st_atim.tv_sec;
      buf->st_atim.tv_nsec = kbuf->st_atim.tv_nsec;
      buf->st_mtim.tv_sec = kbuf->st_mtim.tv_sec;
      buf->st_mtim.tv_nsec = kbuf->st_mtim.tv_nsec;
      buf->st_ctim.tv_sec = kbuf->st_ctim.tv_sec;
      buf->st_ctim.tv_nsec = kbuf->st_ctim.tv_nsec;
      buf->st_blksize = kbuf->st_blksize;
      buf->st_blocks = kbuf->st_blocks;
      /* Check for overflow.  */
      if (buf->st_blocks != kbuf->st_blocks)
	{
	  __set_errno (EOVERFLOW);
	  return -1;
	}
      memset (&buf->st_pad5, 0, sizeof (buf->st_pad5));
      break;

      /* If struct stat64 is different from struct stat then
	 _STAT_VER_KERNEL does not make sense.  */
    case _STAT_VER_KERNEL:
    default:
      __set_errno (EINVAL);
      return -1;
    }

  return 0;
}
#endif /* _MIPS_SIM == _ABIO32 */
