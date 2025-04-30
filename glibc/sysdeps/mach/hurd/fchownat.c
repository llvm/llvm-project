/* Change owner and group of a file relative to open directory.  Hurd version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <stddef.h>
#include <unistd.h>
#include <sys/types.h>
#include <hurd.h>
#include <hurd/fd.h>

/* Change the owner and group of FILE.  */
int
fchownat (int fd, const char *file, uid_t owner, gid_t group, int flag)
{
  error_t err;
  file_t port = __file_name_lookup_at (fd, flag, file, 0, 0);
  if (port == MACH_PORT_NULL)
    return -1;
  err = __file_chown (port, owner, group);
  __mach_port_deallocate (__mach_task_self (), port);
  if (err)
    return __hurd_fail (err);
  return 0;
}
