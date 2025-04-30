/* lutimes -- change access and modification times of a symlink.  Hurd version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <sys/time.h>
#include <errno.h>
#include <stddef.h>
#include <hurd.h>
#include <fcntl.h>

#include "utime-helper.c"

/* Change the access time of FILE to TVP[0] and
   the modification time of FILE to TVP[1].  */
int
__lutimes (const char *file, const struct timeval tvp[2])
{
  error_t err;
  file_t port;

  port = __file_name_lookup (file, O_NOLINK, 0);
  if (port == MACH_PORT_NULL)
    return -1;

  err = hurd_futimes (port, tvp);

  __mach_port_deallocate (__mach_task_self (), port);
  if (err)
    return __hurd_fail (err);
  return 0;
}
weak_alias (__lutimes, lutimes)
