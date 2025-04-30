/* Cleanup function for `struct hurd_port' users.
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
#include <hurd/port.h>

/* The last user of the send right CLEANUP_DATA is now doing
   `longjmp (ENV, VAL)', and this will unwind the frame of
   that last user.  Deallocate the right he will never get back to using.  */

void
_hurd_port_cleanup (void *cleanup_data, jmp_buf env, int val)
{
  __mach_port_deallocate (__mach_task_self (), (mach_port_t) cleanup_data);
}

/* We were cancelled while using a port, and called from the cleanup unwinding.
 */

void
_hurd_port_use_cleanup (void *arg)
{
  struct _hurd_port_use_data *data = arg;

  _hurd_port_free (data->p, &data->link, data->port);
}
