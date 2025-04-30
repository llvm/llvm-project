/* Set a thread's signal state.  Hurd on Mach version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <assert.h>
#include <signal.h>
#include <hurd/signal.h>
#include <hurd/msg.h>

#include <pt-internal.h>

error_t
__pthread_sigstate (struct __pthread *thread, int how,
		    const sigset_t *set, sigset_t *oset, int clear_pending)
{
  error_t err = 0;
  struct hurd_sigstate *ss;
  sigset_t pending;

  ss = _hurd_thread_sigstate (thread->kernel_thread);
  assert (ss);

  _hurd_sigstate_lock (ss);

  if (oset != NULL)
    *oset = ss->blocked;

  if (set != NULL)
    {
      switch (how)
	{
	case SIG_BLOCK:
	  ss->blocked |= *set;
	  break;

	case SIG_SETMASK:
	  ss->blocked = *set;
	  break;

	case SIG_UNBLOCK:
	  ss->blocked &= ~*set;
	  break;

	default:
	  err = EINVAL;
	  break;
	}
      ss->blocked &= ~_SIG_CANT_MASK;
    }

  if (!err && clear_pending)
    __sigemptyset (&ss->pending);

  pending = _hurd_sigstate_pending (ss) & ~ss->blocked;
  _hurd_sigstate_unlock (ss);

  if (!err && pending)
    /* Send a message to the signal thread so it
       will wake up and check for pending signals.  */
    __msg_sig_post (_hurd_msgport, 0, 0, __mach_task_self ());

  return err;
}
