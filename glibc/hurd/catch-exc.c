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

#include <mach/exc_server.h>
#include <hurd/signal.h>
#include <assert.h>

/* Called by the microkernel when a thread gets an exception.  */

kern_return_t
_S_catch_exception_raise (mach_port_t port,
			  thread_t thread,
			  task_t task,
#ifdef EXC_MASK_ALL		/* New interface flavor.  */
			  exception_type_t exception,
			  exception_data_t code,
			  mach_msg_type_number_t codeCnt
#else				/* Vanilla Mach 3.0 interface.  */
			  integer_t exception,
			  integer_t code, integer_t subcode
#endif
			  )
{
  struct hurd_sigstate *ss;
  int signo;
  struct hurd_signal_detail d;

  if (task != __mach_task_self ())
    /* The sender wasn't the kernel.  */
    return EPERM;

  d.exc = exception;
#ifdef EXC_MASK_ALL
  assert (codeCnt >= 2);
  d.exc_code = code[0];
  d.exc_subcode = code[1];
#else
  d.exc_code = code;
  d.exc_subcode = subcode;
#endif

  /* Call the machine-dependent function to translate the Mach exception
     codes into a signal number and subcode.  */
  _hurd_exception2signal (&d, &signo);

  /* Find the sigstate structure for the faulting thread.  */
  __mutex_lock (&_hurd_siglock);
  for (ss = _hurd_sigstates; ss != NULL; ss = ss->next)
    if (ss->thread == thread)
      break;
  __mutex_unlock (&_hurd_siglock);
  if (ss == NULL)
    ss = _hurd_thread_sigstate (thread); /* Allocate a fresh one.  */

  if (__spin_lock_locked (&ss->lock))
    {
      /* Loser.  The thread faulted with its sigstate lock held.  Its
	 sigstate data is now suspect.  So we reset the parts of it which
	 could cause trouble for the signal thread.  Anything else
	 clobbered therein will just hose this user thread, but it's
	 faulting already.

	 This is almost certainly a library bug: unless random memory
	 clobberation caused the sigstate lock to gratuitously appear held,
	 no code should do anything that can fault while holding the
	 sigstate lock.  */

      __spin_unlock (&ss->critical_section_lock);
      ss->context = NULL;
      __spin_unlock (&ss->lock);
    }

  /* Post the signal.  */
  _hurd_internal_post_signal (ss, signo, &d,
			      MACH_PORT_NULL, MACH_MSG_TYPE_PORT_SEND,
			      0);

  return KERN_SUCCESS;
}

#ifdef EXC_MASK_ALL
/* XXX New interface flavor has additional RPCs that we could be using
   instead.  These RPCs roll a thread_get_state/thread_set_state into
   the message, so the signal thread ought to use these to save some calls.
 */
kern_return_t
_S_catch_exception_raise_state (mach_port_t port,
				exception_type_t exception,
				exception_data_t code,
				mach_msg_type_number_t codeCnt,
				int *flavor,
				thread_state_t old_state,
				mach_msg_type_number_t old_stateCnt,
				thread_state_t new_state,
				mach_msg_type_number_t *new_stateCnt)
{
  abort ();
  return KERN_FAILURE;
}

kern_return_t
_S_catch_exception_raise_state_identity (mach_port_t exception_port,
					 thread_t thread,
					 task_t task,
					 exception_type_t exception,
					 exception_data_t code,
					 mach_msg_type_number_t codeCnt,
					 int *flavor,
					 thread_state_t old_state,
					 mach_msg_type_number_t old_stateCnt,
					 thread_state_t new_state,
					 mach_msg_type_number_t *new_stateCnt)
{
  abort ();
  return KERN_FAILURE;
}
#endif
