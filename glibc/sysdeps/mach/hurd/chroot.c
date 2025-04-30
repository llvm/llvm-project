/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <unistd.h>

#include <hurd.h>
#include <hurd/port.h>

/* Make PATH be the root directory (the starting point for absolute
   paths).  Note that while on traditional UNIX systems this call is
   restricted to the super-user, it isn't on the Hurd.  */
int
chroot (const char *path)
{
  const char *lookup;
  size_t len;
  file_t dir, root;
  error_t err;

  /* Append trailing "/." to directory name to force ENOTDIR if it's not a
     directory and EACCES if we don't have search permission.  */
  len = strlen (path);
  if (len >= 2 && path[len - 2] == '/' && path[len - 1] == '.')
    lookup = path;
  else if (len == 0)
    /* Special-case empty file name according to POSIX.  */
    return __hurd_fail (ENOENT);
  else
    {
      char *n = alloca (len + 3);
      memcpy (n, path, len);
      n[len] = '/';
      n[len + 1] = '.';
      n[len + 2] = '\0';
      lookup = n;
    }

  dir = __file_name_lookup (lookup, 0, 0);
  if (dir == MACH_PORT_NULL)
    return -1;

  /* Prevent going through DIR's ..  */
  err = __file_reparent (dir, MACH_PORT_NULL, &root);
  __mach_port_deallocate (__mach_task_self (), dir);
  if (err)
    return __hurd_fail (err);

  _hurd_port_set (&_hurd_ports[INIT_PORT_CRDIR], root);
  return 0;
}
