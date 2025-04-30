/* lchmod -- Change the protections of a file or symbolic link.  Hurd version.
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

#include <errno.h>
#include <stddef.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <hurd.h>

/* Change the protections of FILE to MODE.  */
int
lchmod (const char *file, mode_t mode)
{
  error_t err;
  file_t port = __file_name_lookup (file, O_NOLINK, 0);
  if (port == MACH_PORT_NULL)
    return -1;
  err = __file_chmod (port, mode);
  __mach_port_deallocate (__mach_task_self (), port);
  if (err)
    return __hurd_fail (err);
  return 0;
}
