/* Generic implementation of statx based on fstatat64.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

/* Obtain the original definition of struct statx.  */
#undef __statx_defined
#define statx original_statx
#include <bits/types/struct_statx.h>
#undef statx

static inline struct statx_timestamp
statx_convert_timestamp (struct timespec tv)
{
  return (struct statx_timestamp) { tv.tv_sec, tv.tv_nsec };
}

/* Approximate emulation of statx.  This will always fill in
   POSIX-mandated attributes even if the underlying file system does
   not actually support it (for example, GID and UID on file systems
   without UNIX-style permissions).  */
static __attribute__ ((unused)) int
statx_generic (int fd, const char *path, int flags,
               unsigned int mask, struct statx *buf)
{
  /* Flags which need to be cleared before passing them to
     fstatat64.  */
  static const int clear_flags = AT_STATX_SYNC_AS_STAT;

    /* Flags supported by our emulation.  */
  static const int supported_flags
    = AT_EMPTY_PATH | AT_NO_AUTOMOUNT | AT_SYMLINK_NOFOLLOW | clear_flags;

  if (__glibc_unlikely ((flags & ~supported_flags) != 0))
    {
      __set_errno (EINVAL);
      return -1;
    }

  struct stat64 st;
  int ret = __fstatat64 (fd, path, &st, flags & ~clear_flags);
  if (ret != 0)
    return ret;

  /* The interface is defined in such a way that unused (padding)
     fields have to be cleared.  STATX_BASIC_STATS corresponds to the
     data which is available via fstatat64.  */
  struct original_statx obuf =
    {
      .stx_mask = STATX_BASIC_STATS,
      .stx_blksize = st.st_blksize,
      .stx_nlink = st.st_nlink,
      .stx_uid = st.st_uid,
      .stx_gid = st.st_gid,
      .stx_mode = st.st_mode,
      .stx_ino = st.st_ino,
      .stx_size = st.st_size,
      .stx_blocks = st.st_blocks,
      .stx_atime = statx_convert_timestamp (st.st_atim),
      .stx_ctime = statx_convert_timestamp (st.st_ctim),
      .stx_mtime = statx_convert_timestamp (st.st_mtim),
      .stx_rdev_major = major (st.st_rdev),
      .stx_rdev_minor = minor (st.st_rdev),
      .stx_dev_major = major (st.st_dev),
      .stx_dev_minor = minor (st.st_dev),
    };
  _Static_assert (sizeof (*buf) >= sizeof (obuf), "struct statx size");
  memcpy (buf, &obuf, sizeof (obuf));

  return 0;
}
