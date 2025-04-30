/* Convert between `struct statfs' format, and `struct statfs64' format.
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

#include <sys/statfs.h>
#include <errno.h>

static inline int
statfs64_conv (struct statfs *buf, const struct statfs64 *buf64)
{
# define DO(memb)							      \
  buf->memb = buf64->memb;						      \
  if (sizeof buf->memb != sizeof buf64->memb && buf->memb != buf64->memb)     \
    {									      \
      __set_errno (EOVERFLOW);						      \
      return -1;							      \
    }

  DO (f_type);
  DO (f_bsize);
  DO (f_blocks);
  DO (f_bfree);
  DO (f_bavail);
  DO (f_files);
  DO (f_fsid);
  DO (f_namelen);
  DO (f_favail);
  DO (f_frsize);
  DO (f_flag);

# undef DO

  return 0;
}
