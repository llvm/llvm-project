/* Setup thread stack.  Hurd/i386 version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <stdint.h>
#include <assert.h>
#include <mach.h>

#include <pt-internal.h>

/* The stack layout used on the i386 is:

    -----------------
   |  ARG            |
    -----------------
   |  START_ROUTINE  |
    -----------------
   |  0              |
    -----------------
 */

/* Set up the stack for THREAD, such that it appears as if
   START_ROUTINE and ARG were passed to the new thread's entry-point.
   Return the stack pointer for the new thread.  */
static void *
stack_setup (struct __pthread *thread,
	     void *(*start_routine) (void *), void *arg)
{
  error_t err;
  uintptr_t *bottom, *top;

  /* Calculate the top of the new stack.  */
  bottom = thread->stackaddr;
  top = (uintptr_t *) ((uintptr_t) bottom + thread->stacksize
		       + ((thread->guardsize + __vm_page_size - 1)
			  / __vm_page_size) * __vm_page_size);

  if (start_routine != NULL)
    {
      /* And then the call frame.  */
      top -= 3;
      top = (uintptr_t *) ((uintptr_t) top & ~0xf);
      top[2] = (uintptr_t) arg;	/* Argument to START_ROUTINE.  */
      top[1] = (uintptr_t) start_routine;
      top[0] = (uintptr_t) thread;
      *--top = 0;		/* Fake return address.  */
    }

  if (thread->guardsize)
    {
      err = __vm_protect (__mach_task_self (), (vm_address_t) bottom,
			  thread->guardsize, 0, 0);
      assert_perror (err);
    }

  return top;
}

int
__pthread_setup (struct __pthread *thread,
		 void (*entry_point) (struct __pthread *, void *(*)(void *),
				      void *), void *(*start_routine) (void *),
		 void *arg)
{
  tcbhead_t *tcb;
  error_t err;
  mach_port_t ktid;

  thread->mcontext.pc = entry_point;
  thread->mcontext.sp = stack_setup (thread, start_routine, arg);

  ktid = __mach_thread_self ();
  if (thread->kernel_thread == ktid)
    /* Fix up the TCB for the main thread.  The C library has already
       installed a TCB, which we want to keep using.  This TCB must not
       be freed so don't register it in the thread structure.  On the
       other hand, it's not yet possible to reliably release a TCB.
       Leave the unused one registered so that it doesn't leak.  The
       only thing left to do is to correctly set the `self' member in
       the already existing TCB.  */
    tcb = THREAD_SELF;
  else
    {
      err = __thread_set_pcsptp (thread->kernel_thread,
				 1, thread->mcontext.pc,
				 1, thread->mcontext.sp,
				 1, thread->tcb);
      assert_perror (err);
      tcb = thread->tcb;
    }
  __mach_port_deallocate (__mach_task_self (), ktid);

  tcb->self = thread->kernel_thread;

  return 0;
}
