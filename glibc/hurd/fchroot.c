/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/port.h>

/* Change the current root directory to FD.  */
int
fchroot (int fd)
{
  error_t err;
  file_t dir;

  err = HURD_DPORT_USE (fd,
			({
			  dir = __file_name_lookup_under (port, ".", 0, 0);
			  dir == MACH_PORT_NULL ? errno : 0;
			}));

  if (! err)
    {
      file_t root;

      /* Prevent going through DIR's ..  */
      err = __file_reparent (dir, MACH_PORT_NULL, &root);
      __mach_port_deallocate (__mach_task_self (), dir);
      if (! err)
	_hurd_port_set (&_hurd_ports[INIT_PORT_CRDIR], root);
    }

  return err ? __hurd_fail (err) : 0;
}
