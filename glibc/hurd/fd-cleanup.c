/* Cleanup function for `struct hurd_fd' users.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <mach.h>
#include <hurd/fd.h>

/* We were cancelled while using an fd, and called from the cleanup unwinding.
 */

void
_hurd_fd_port_use_cleanup (void *arg)
{
  struct _hurd_fd_port_use_data *data = arg;

  _hurd_port_free (&data->d->port, &data->ulink, data->port);
  if (data->ctty != MACH_PORT_NULL)
    _hurd_port_free (&data->d->ctty, &data->ctty_ulink, data->ctty);
}
