/* Create a device file relative to an open directory.  Hurd version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <hurd.h>
#include <shlib-compat.h>

#if SHLIB_COMPAT(libc, GLIBC_2_4, GLIBC_2_33)
/* Create a device file named PATH relative to FD, with permission and
   special bits MODE and device number DEV (which can be constructed
   from major and minor device numbers with the `makedev' macro
   above).  */
int
__xmknodat (int vers, int fd, const char *path, mode_t mode, dev_t *dev)
{
  if (vers != _MKNOD_VER)
    return __hurd_fail (EINVAL);

  return __mknodat (fd, path, mode, *dev);
}
#endif
