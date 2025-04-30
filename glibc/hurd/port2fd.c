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

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/signal.h>
#include <hurd/term.h>
#include <fcntl.h>

/* Store PORT in file descriptor D, doing appropriate ctty magic.
   FLAGS are as for `open'; only O_IGNORE_CTTY and O_CLOEXEC are meaningful.
   D should be locked, and will not be unlocked.  */

void
_hurd_port2fd (struct hurd_fd *d, io_t dport, int flags)
{
  mach_port_t cttyid;
  io_t ctty = MACH_PORT_NULL;

  if (!(flags & O_IGNORE_CTTY))
    __USEPORT (CTTYID,
	       ({
		 if (port != MACH_PORT_NULL /* Do we have a ctty? */
		     && ! __term_getctty (dport, &cttyid))
		   /* Could this be it? */
		   {
		     __mach_port_deallocate (__mach_task_self (), cttyid);
		     /* This port is capable of being a controlling tty.
			Is it ours?  */
		     if (cttyid == port)
		       __term_open_ctty (dport, _hurd_pid, _hurd_pgrp, &ctty);
		     /* XXX if this port is our ctty, but we are not doing
			ctty style i/o because term_become_ctty barfed,
			what to do?  */
		   }
		 0;
	       }));

  /* Install PORT in the descriptor cell, leaving it locked.  */
  {
    mach_port_t old
      = _hurd_userlink_clear (&d->port.users) ? d->port.port : MACH_PORT_NULL;
    d->port.port = dport;
    d->flags = (flags & O_CLOEXEC) ? FD_CLOEXEC : 0;
    if (old != MACH_PORT_NULL)
      __mach_port_deallocate (__mach_task_self (), old);
  }

  _hurd_port_set (&d->ctty, ctty);
}
