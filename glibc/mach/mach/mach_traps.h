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

/* Declare the few Mach system calls (except mach_msg, in <mach/message.h>).
   This does not include the kernel RPC shortcut calls (in <mach-shortcuts.h>).
   */

#ifndef	_MACH_MACH_TRAPS_H

#define _MACH_MACH_TRAPS_H_	1

#include <mach/port.h>
#include <mach/message.h>	/* mach_msg_timeout_t */
#include <mach/kern_return.h>

/* Create and return a new receive right.  */
extern mach_port_t mach_reply_port (void);

/* Return the thread control port for the calling thread.  */
extern mach_port_t mach_thread_self (void);

/* Return the task control port for the calling task.
   The parens are needed to protect against the macro in <mach_init.h>.  */
extern mach_port_t (mach_task_self) (void);

/* Return the host information port for the host of the calling task.
   The parens are needed to protect against the macro in <mach_init.h>.  */
extern mach_port_t (mach_host_self) (void);

/* Attempt to context switch the current thread off the processor.  Returns
   true if there are other threads that can be run and false if not.  */
extern boolean_t swtch (void);

/* Attempt to context switch the current thread off the processor.  Lower
   the thread's priority as much as possible.  The thread's priority will
   be restored when it runs again.  PRIORITY is currently unused.  Return
   true if there are other threads that can be run and false if not.  */
extern boolean_t swtch_pri (int priority);

/* Attempt to context switch the current thread off the processor.  Try
   to run NEW_THREAD next, ignoring normal scheduling policies.  The
   OPTION value comes from <mach/thread_switch.h>.  If OPTION is
   SWITCH_OPTION_WAIT, then block the current thread for TIME
   milliseconds.  If OPTION is SWITCH_OPTION_DEPRESS, then block for
   TIME milliseconds and depress the thread's priority as done by
   swtch_pri.  If OPTION is SWITCH_OPTION_NONE, ignore TIME.  */
kern_return_t thread_switch (mach_port_t new_thread,
			     int option, mach_msg_timeout_t option_time);

/* Block the current thread until the kernel (or device) event
   identified by EVENT occurs.  */
kern_return_t evc_wait (unsigned int event);

/* Display a null-terminated character string on the Mach console. This
   system call is meant as a debugging tool useful to circumvent messaging
   altogether.  */

extern void mach_print(const char *s);

#endif	/* mach/mach_traps.h */
