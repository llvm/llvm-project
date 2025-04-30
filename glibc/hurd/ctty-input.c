/* _hurd_ctty_input -- Do an input RPC and generate SIGTTIN if necessary.
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

#include <hurd.h>
#include <hurd/signal.h>

/* Call *RPC on PORT and/or CTTY.  If a call on CTTY returns EBACKGROUND,
   generate SIGTTIN or EIO as appropriate.  */

error_t
_hurd_ctty_input (io_t port, io_t ctty, error_t (*rpc) (io_t))
{
  error_t err;

  if (ctty == MACH_PORT_NULL)
    return (*rpc) (port);

  do
    {
      err = (*rpc) (ctty);
      if (err == EBACKGROUND)
	{
	  /* We are a background job and tried to read from the tty.
	     We should probably get a SIGTTIN signal.  */
	  if (_hurd_orphaned)
	    /* Our process group is orphaned.  Don't stop; just fail.  */
	    err = EIO;
	  else
	    {
	      struct hurd_sigstate *ss = _hurd_self_sigstate ();
	      struct sigaction *actions;

	      _hurd_sigstate_lock (ss);
	      actions = _hurd_sigstate_actions (ss);
	      if (__sigismember (&ss->blocked, SIGTTIN)
		  || actions[SIGTTIN].sa_handler == SIG_IGN)
		/* We are blocking or ignoring SIGTTIN.  Just fail.  */
		err = EIO;
	      _hurd_sigstate_unlock (ss);

	      if (err == EBACKGROUND)
		{
		  /* Send a SIGTTIN signal to our process group.

		     We must remember here not to clobber ERR, since
		     the loop condition below uses it to recall that
		  we should retry after a stop.  */

		  __USEPORT (CTTYID, _hurd_sig_post (0, SIGTTIN, port));
		  /* XXX what to do if error here? */

		  /* At this point we should have just run the handler for
		     SIGTTIN or resumed after being stopped.  Now this is
		     still a "system call", so check to see if we should
		  restart it.  */
		  _hurd_sigstate_lock (ss);
		  actions = _hurd_sigstate_actions (ss);
		  if (!(actions[SIGTTIN].sa_flags & SA_RESTART))
		    err = EINTR;
		  _hurd_sigstate_unlock (ss);
		}
	    }
	}
      /* If the last RPC generated a SIGTTIN, loop to try it again.  */
    } while (err == EBACKGROUND);

  return err;
}
