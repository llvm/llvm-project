/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <string.h>

error_t
_hurd_fd_read (struct hurd_fd *fd, void *buf, size_t *nbytes, loff_t offset)
{
  error_t err;
  char *data;
  mach_msg_type_number_t nread;

  error_t readfd (io_t port)
    {
      return __io_read (port, &data, &nread, offset, *nbytes);
    }

  data = buf;
  nread = *nbytes;
  if (err = HURD_FD_PORT_USE_CANCEL (fd, _hurd_ctty_input (port, ctty, readfd)))
    return err;

  if (data != buf)
    {
      if (nread > *nbytes)	/* Sanity check for bogus server.  */
	{
	  __vm_deallocate (__mach_task_self (), (vm_address_t) data, nread);
	  return EGRATUITOUS;
	}
      memcpy (buf, data, nread);
      __vm_deallocate (__mach_task_self (), (vm_address_t) data, nread);
    }

  *nbytes = nread;
  return 0;
}
