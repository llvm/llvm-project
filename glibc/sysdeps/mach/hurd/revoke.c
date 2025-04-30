/* Revoke the access of all descriptors currently open on a file.  Hurd version
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <errno.h>
#include <hurd.h>

int
__revoke (const char *file_name)
{
  error_t err;
  file_t file = __file_name_lookup (file_name, 0, 0);

  if (file == MACH_PORT_NULL)
    return -1;

  err = __io_revoke (file);
  __mach_port_deallocate (__mach_task_self (), file);

  if (err)
    return __hurd_fail (err);
  return 0;
}

weak_alias (__revoke, revoke)
