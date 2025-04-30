/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/types.h>
#include <signal.h>
#include <hurd.h>
#include <hurd/port.h>
#include <hurd/signal.h>
#include <hurd/msg.h>

/* Send a `sig_post' RPC to process number PID.  If PID is zero,
   send the message to all processes in the current process's process group.
   If PID is < -1, send SIG to all processes in process group - PID.
   SIG and REFPORT are passed along in the request message.  */
error_t
_hurd_sig_post (pid_t pid, int sig, mach_port_t arg_refport)
{
  int delivered = 0;		/* Set when we deliver any signal.  */
  error_t err;
  mach_port_t proc;
  struct hurd_userlink ulink;

  inline void kill_pid (pid_t pid) /* Kill one PID.  */
    {
      err = HURD_MSGPORT_RPC (__proc_getmsgport (proc, pid, &msgport),
			      (refport = arg_refport, 0), 0,
			      /* If no message port we cannot send signals.  */
			      msgport == MACH_PORT_NULL ? EPERM
			      : __msg_sig_post (msgport, sig, 0, refport));
      if (! err)
	delivered = 1;
    }

  proc = _hurd_port_get (&_hurd_ports[INIT_PORT_PROC], &ulink);

  if (pid <= 0)
    {
      /* Send SIG to each process in pgrp (- PID).  */
      mach_msg_type_number_t npids = 10, i;
      pid_t pidsbuf[10], *pids = pidsbuf;

      err = __proc_getpgrppids (proc, - pid, &pids, &npids);
      if (!err)
	{
	  int self = 0;
	  for (i = 0; i < npids; ++i)
	    if (pids[i] == _hurd_pid)
	      /* We must do ourselves last so we are not suspended
		 and fail to suspend the other processes in the pgrp.  */
	      self = 1;
	    else
	      {
		kill_pid (pids[i]);
		if (err == ESRCH)
		  /* The process died already.  Ignore it.  */
		  err = 0;
	      }
	  if (pids != pidsbuf)
	    __vm_deallocate (__mach_task_self (),
			     (vm_address_t) pids, npids * sizeof (pids[0]));

	  if (self)
	    kill_pid (_hurd_pid);
	}
    }
  else
    kill_pid (pid);

  _hurd_port_free (&_hurd_ports[INIT_PORT_PROC], &ulink, proc);

  /* If we delivered no signals, but ERR is clear, this must mean that
     every kill_pid call failed with ESRCH, meaning all the processes in
     the pgrp died between proc_getpgrppids and kill_pid; in that case we
     fail with ESRCH.  */
  return delivered ? 0 : err ?: ESRCH;
}
weak_alias (_hurd_sig_post, hurd_sig_post)
