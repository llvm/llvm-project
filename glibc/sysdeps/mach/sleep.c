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

#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <mach.h>
#include <sysdep-cancel.h>

/* Make the process sleep for SECONDS seconds, or until a signal arrives
   and is not ignored.  The function returns the number of seconds less
   than SECONDS which it actually slept (zero if it slept the full time).
   There is no return value to indicate error, but if `sleep' returns
   SECONDS, it probably didn't work.  */
unsigned int
__sleep (unsigned int seconds)
{
  time_t before, after;
  mach_port_t recv;
  int cancel_oldtype;

  recv = __mach_reply_port ();

  before = time_now ();
  cancel_oldtype = LIBC_CANCEL_ASYNC();
  (void) __mach_msg (NULL, MACH_RCV_MSG|MACH_RCV_TIMEOUT|MACH_RCV_INTERRUPT,
		     0, 0, recv, seconds * 1000, MACH_PORT_NULL);
  LIBC_CANCEL_RESET (cancel_oldtype);
  after = time_now ();
  __mach_port_destroy (__mach_task_self (), recv);

  return seconds - (after - before);
}
weak_alias (__sleep, sleep)
