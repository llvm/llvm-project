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
#include <hurd/signal.h>
#include <hurd/msg.h>
#include <setjmp.h>

/* Handle signal SIGNO in the calling thread.
   If SS is not NULL it is the sigstate for the calling thread;
   SS->lock is held on entry and released before return.  */

int
_hurd_raise_signal (struct hurd_sigstate *ss,
		    int signo, const struct hurd_signal_detail *detail)
{
  if (signo <= 0 || signo >= NSIG)
    {
      if (ss)
	__spin_unlock (&ss->lock);
      return EINVAL;
    }

  if (ss == NULL)
    {
      ss = _hurd_self_sigstate ();
      __spin_lock (&ss->lock);
    }

  /* Mark SIGNO as pending to be delivered.  */
  __sigaddset (&ss->pending, signo);
  ss->pending_data[signo] = *detail;

  __spin_unlock (&ss->lock);

  /* Send a message to the signal thread so it will wake up and check for
     pending signals.  This is a generic "poll request" message (SIGNO==0)
     rather than delivering this signal and its detail, because we have
     already marked the signal as pending for the particular thread we
     want.  Generating the signal with an RPC might deliver it to some
     other thread.  */
  return __msg_sig_post (_hurd_msgport, 0, 0, __mach_task_self ());
}
libc_hidden_def (_hurd_raise_signal)
