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
#include <hurd.h>
#include <hurd/signal.h>
#include <hurd/msg.h>
#include <sysdep-cancel.h>

/* Change the set of blocked signals to SET,
   wait until a signal arrives, and restore the set of blocked signals.  */
int
__sigsuspend (const sigset_t *set)
{
  struct hurd_sigstate *ss;
  sigset_t newmask, oldmask, pending;
  mach_port_t wait;
  mach_msg_header_t msg;
  int cancel_oldtype;

  if (set != NULL)
    /* Crash before locking.  */
    newmask = *set;

  /* Get a fresh port we will wait on.  */
  wait = __mach_reply_port ();

  ss = _hurd_self_sigstate ();

  _hurd_sigstate_lock (ss);

  oldmask = ss->blocked;
  if (set != NULL)
    /* Change to the new blocked signal mask.  */
    ss->blocked = newmask & ~_SIG_CANT_MASK;

  /* Notice if any pending signals just became unblocked.  */
  pending = _hurd_sigstate_pending (ss) & ~ss->blocked;

  /* Tell the signal thread to message us when a signal arrives.  */
  ss->suspended = wait;
  _hurd_sigstate_unlock (ss);

  if (pending)
    /* Tell the signal thread to check for pending signals.  */
    __msg_sig_post (_hurd_msgport, 0, 0, __mach_task_self ());

  /* Wait for the signal thread's message.  */

  cancel_oldtype = LIBC_CANCEL_ASYNC();
  __mach_msg (&msg, MACH_RCV_MSG, 0, sizeof (msg), wait,
	      MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
  LIBC_CANCEL_RESET (cancel_oldtype);
  __mach_port_destroy (__mach_task_self (), wait);

  /* Restore the old mask and check for pending signals again.  */
  _hurd_sigstate_lock (ss);
  ss->blocked = oldmask;
  pending = _hurd_sigstate_pending(ss) & ~ss->blocked;
  _hurd_sigstate_unlock (ss);

  if (pending)
    /* Tell the signal thread to check for pending signals.  */
    __msg_sig_post (_hurd_msgport, 0, 0, __mach_task_self ());

  /* We've been interrupted!  And a good thing, too.
     Otherwise we'd never return.
     That's right; this function always returns an error.  */
  errno = EINTR;
  return -1;
}
libc_hidden_def (__sigsuspend)
weak_alias (__sigsuspend, sigsuspend)
