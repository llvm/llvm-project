/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <hurd/fd.h>

error_t
_hurd_fd_close (struct hurd_fd *fd)
{
  error_t err;

  HURD_CRITICAL_BEGIN;

  __spin_lock (&fd->port.lock);
  if (fd->port.port == MACH_PORT_NULL)
    {
      __spin_unlock (&fd->port.lock);
      err = EBADF;
    }
  else
    {
      /* Clear the descriptor's port cells.
	 This deallocates the ports if noone else is still using them.  */
      _hurd_port_set (&fd->ctty, MACH_PORT_NULL);
      _hurd_port_locked_set (&fd->port, MACH_PORT_NULL);
      err = 0;
    }

  HURD_CRITICAL_END;

  return err;
}
