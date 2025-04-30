/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <mach.h>
#include <sys/time.h>
#include <unistd.h>
#include <sysdep-cancel.h>

/* Sleep USECONDS microseconds, or until a previously set timer goes off.  */
int
usleep (useconds_t useconds)
{
  mach_port_t recv;
  int cancel_oldtype;
  mach_msg_timeout_t timeout = useconds / 1000 + (useconds % 1000 != 0);

  recv = __mach_reply_port ();

  cancel_oldtype = LIBC_CANCEL_ASYNC();
  (void) __mach_msg (NULL, MACH_RCV_MSG|MACH_RCV_TIMEOUT|MACH_RCV_INTERRUPT,
		     0, 0, recv, timeout, MACH_PORT_NULL);
  LIBC_CANCEL_RESET (cancel_oldtype);
  __mach_port_destroy (mach_task_self (), recv);

  return 0;
}
