/* unlinkat -- Remove a name relative to an open directory.  Hurd version.
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
#include <hurd.h>
#include <hurd/fd.h>


/* Remove the link named NAME.  */
int
unlinkat (int fd, const char *name, int flag)
{
  error_t err;
  file_t dir;
  const char *file;

  if ((flag &~ AT_REMOVEDIR) != 0)
    {
      __set_errno (EINVAL);
      return -1;
    }

  dir = __directory_name_split_at (fd, name, (char **) &file);
  if (dir == MACH_PORT_NULL)
    return -1;

  err = ((flag & AT_REMOVEDIR) ? __dir_rmdir : __dir_unlink) (dir, file);
  __mach_port_deallocate (__mach_task_self (), dir);

  if (err)
    return __hurd_fail (err);
  return 0;
}
