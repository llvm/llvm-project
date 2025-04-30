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
#include <signal.h>
#include <hurd.h>
#include <hurd/signal.h>

/* If ACT is not NULL, change the action for SIG to *ACT.
   If OACT is not NULL, put the old action for SIG in *OACT.  */
int
__libc_sigaction (int sig, const struct sigaction *act,
		  struct sigaction *oact)
{
  struct hurd_sigstate *ss;
  struct sigaction a, old;
  sigset_t pending;

  if (act != NULL && act->sa_handler != SIG_DFL
      && ((__sigmask (sig) & _SIG_CANT_MASK) || act->sa_handler == SIG_ERR))
    {
      errno = EINVAL;
      return -1;
    }

  /* Copy so we fault before taking locks.  */
  if (act != NULL)
    a = *act;

  ss = _hurd_self_sigstate ();

  __spin_lock (&ss->critical_section_lock);
  _hurd_sigstate_lock (ss);
  old = _hurd_sigstate_actions (ss) [sig];
  if (act != NULL)
    _hurd_sigstate_actions (ss) [sig] = a;

  if (act != NULL && sig == SIGCHLD
      && (a.sa_flags & SA_NOCLDSTOP) != (old.sa_flags & SA_NOCLDSTOP))
    {
      _hurd_sigstate_unlock (ss);

      /* Inform the proc server whether or not it should send us SIGCHLD for
	 stopped children.  We do this in a critical section so that no
	 SIGCHLD can arrive in the middle and be of indeterminate status.  */
      __USEPORT (PROC,
		 __proc_mod_stopchild (port, !(a.sa_flags & SA_NOCLDSTOP)));

      _hurd_sigstate_lock (ss);
      pending = _hurd_sigstate_pending (ss) & ~ss->blocked;
    }
  else if (act != NULL && (a.sa_handler == SIG_IGN || a.sa_handler == SIG_DFL))
    /* We are changing to an action that might be to ignore SIG signals.
       If SIG is blocked and pending and the new action is to ignore it, we
       must remove it from the pending set now; if the action is changed
       back and then SIG is unblocked, the signal pending now should not
       arrive.  So wake up the signal thread to check the new state and do
       the right thing.  */
    pending = _hurd_sigstate_pending (ss) & __sigmask (sig);
  else
    pending = 0;

  _hurd_sigstate_unlock (ss);
  __spin_unlock (&ss->critical_section_lock);

  if (pending)
    __msg_sig_post (_hurd_msgport, 0, 0, __mach_task_self ());

  if (oact != NULL)
    *oact = old;

  return 0;
}
libc_hidden_def (__libc_sigaction)
