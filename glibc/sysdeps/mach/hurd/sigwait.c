/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <hurd/sigpreempt.h>
#include <assert.h>
#include <sysdep-cancel.h>

/* Select any of pending signals from SET or wait for any to arrive.  */
int
__sigwait (const sigset_t *set, int *sig)
{
  struct hurd_sigstate *ss;
  sigset_t mask, ready, blocked;
  int signo = 0;
  struct hurd_signal_preemptor preemptor;
  jmp_buf buf;
  mach_port_t wait;
  mach_msg_header_t msg;
  int cancel_oldtype;

  sighandler_t
    preempt_fun (struct hurd_signal_preemptor *pe,
		 struct hurd_sigstate *ss,
		 int *sigp,
		 struct hurd_signal_detail *detail)
    {
      if (signo)
	/* We've already been run; don't interfere. */
	return SIG_ERR;

      signo = *sigp;

      /* Make sure this is all kosher */
      assert (__sigismember (&mask, signo));

      /* Restore the blocking mask. */
      ss->blocked = blocked;

      return pe->handler;
    }

  void
    handler (int sig)
    {
      assert (sig == signo);
      longjmp (buf, 1);
    }

  wait = __mach_reply_port ();

  if (set != NULL)
    /* Crash before locking */
    mask = *set;
  else
    __sigemptyset (&mask);

  ss = _hurd_self_sigstate ();
  cancel_oldtype = LIBC_CANCEL_ASYNC();
  _hurd_sigstate_lock (ss);

  /* See if one of these signals is currently pending.  */
  sigset_t pending = _hurd_sigstate_pending (ss);
  __sigandset (&ready, &pending, &mask);
  if (! __sigisemptyset (&ready))
    {
      for (signo = 1; signo < NSIG; signo++)
	if (__sigismember (&ready, signo))
	  {
	    __sigdelset (&ready, signo);
	    goto all_done;
	  }
      /* Huh?  Where'd it go? */
      abort ();
    }

  /* Wait for one of them to show up.  */

  if (!setjmp (buf))
    {
      /* Make the preemptor */
      preemptor.signals = mask;
      preemptor.first = 0;
      preemptor.last = -1;
      preemptor.preemptor = preempt_fun;
      preemptor.handler = handler;

      /* Install this preemptor */
      preemptor.next = ss->preemptors;
      ss->preemptors = &preemptor;

      /* Unblock the expected signals */
      blocked = ss->blocked;
      ss->blocked &= ~mask;

      _hurd_sigstate_unlock (ss);

      /* Wait. */
      __mach_msg (&msg, MACH_RCV_MSG, 0, sizeof (msg), wait,
		  MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
      abort ();
    }
  else
    {
      assert (signo);

      _hurd_sigstate_lock (ss);

      /* Delete our preemptor. */
      assert (ss->preemptors == &preemptor);
      ss->preemptors = preemptor.next;
    }


all_done:
  _hurd_sigstate_unlock (ss);
  LIBC_CANCEL_RESET (cancel_oldtype);

  __mach_port_destroy (__mach_task_self (), wait);
  *sig = signo;
  return 0;
}

libc_hidden_def (__sigwait)
weak_alias (__sigwait, sigwait)
