/* Thread cancellation support.
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

#include <hurd/signal.h>
#include <hurd/interrupt.h>
#include <assert.h>
#include <thread_state.h>


/* See hurdsig.c.  */
extern mach_port_t _hurdsig_abort_rpcs (struct hurd_sigstate *ss,
					int signo, int sigthread,
					struct machine_thread_all_state *,
					int *state_change,
					mach_port_t *reply_port,
					mach_msg_type_name_t reply_port_type,
					int untraced);

error_t
hurd_thread_cancel (thread_t thread)
{
  struct hurd_sigstate *ss = _hurd_thread_sigstate (thread);
  struct machine_thread_all_state state;
  int state_change;
  error_t err;

  if (! ss)
    return EINVAL;
  if (ss == _hurd_self_sigstate ())
    {
      /* We are cancelling ourselves, so it is easy to succeed
	 quickly.  Since this function is not a cancellation point, we
	 just leave the flag set pending the next cancellation point
	 (hurd_check_cancel or RPC) and return success.  */
      ss->cancel = 1;
      return 0;
    }

  assert (! __spin_lock_locked (&ss->critical_section_lock));
  __spin_lock (&ss->critical_section_lock);
  __spin_lock (&ss->lock);
  err = __thread_suspend (thread);
  __spin_unlock (&ss->lock);

  if (! err)
    {
      /* Set the flag telling the thread its operation is being cancelled.  */
      ss->cancel = 1;

      /* Interrupt any interruptible RPC now in progress.  */
      state.set = 0;
      _hurdsig_abort_rpcs (ss, 0, 0, &state, &state_change, NULL, 0, 0);
      if (state_change)
	err = __thread_set_state (thread, MACHINE_THREAD_STATE_FLAVOR,
				  (natural_t *) &state.basic,
				  MACHINE_THREAD_STATE_COUNT);

      if (ss->cancel_hook)
	/* The code being cancelled has a special wakeup function.
	   Calling this should make the thread wake up and check the
	   cancellation flag.  */
	(*ss->cancel_hook) ();

      __thread_resume (thread);
    }

  _hurd_critical_section_unlock (ss);
  return err;
}


int
hurd_check_cancel (void)
{
  struct hurd_sigstate *ss = _hurd_self_sigstate ();
  int cancel;

  __spin_lock (&ss->lock);
  assert (! __spin_lock_locked (&ss->critical_section_lock));
  cancel = ss->cancel;
  ss->cancel = 0;
  __spin_unlock (&ss->lock);

  return cancel;
}
