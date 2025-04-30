/* Deallocate the kernel thread resources.  Mach version.
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

#include <assert.h>
#include <errno.h>
#include <mach.h>

#include <mach/mig_support.h>

#include <pt-internal.h>

/* Terminate the kernel thread associated with THREAD, and deallocate its
   right reference and its stack.  The function also drops a reference
   on THREAD.  */
void
__pthread_thread_terminate (struct __pthread *thread)
{
  thread_t kernel_thread, self_ktid;
  mach_port_t wakeup_port, reply_port;
  void *stackaddr;
  size_t stacksize;
  error_t err;

  kernel_thread = thread->kernel_thread;

  if (thread->stack)
    {
      stackaddr = thread->stackaddr;
      stacksize = ((thread->guardsize + __vm_page_size - 1)
		   / __vm_page_size) * __vm_page_size + thread->stacksize;
    }
  else
    {
      stackaddr = NULL;
      stacksize = 0;
    }

  wakeup_port = thread->wakeupmsg.msgh_remote_port;

  /* Each thread has its own reply port, allocated from MiG stub code calling
     __mig_get_reply_port.  Destroying it is a bit tricky because the calls
     involved are also RPCs, causing the creation of a new reply port if
     currently null. The __thread_terminate_release call is actually a one way
     simple routine designed not to require a reply port.  */
  self_ktid = __mach_thread_self ();
  reply_port = (self_ktid == kernel_thread)
      ? __mig_get_reply_port () : MACH_PORT_NULL;
  __mach_port_deallocate (__mach_task_self (), self_ktid);

  /* Finally done with the thread structure.  */
  __pthread_dealloc (thread);

  /* The wake up port is now no longer needed.  */
  __mach_port_destroy (__mach_task_self (), wakeup_port);

  /* Terminate and release all that's left.  */
  err = __thread_terminate_release (kernel_thread, mach_task_self (),
				    kernel_thread, reply_port,
				    (vm_address_t) stackaddr, stacksize);

  /* The kernel does not support it yet.  Leak but at least terminate
     correctly.  */
  err = __thread_terminate (kernel_thread);

  /* We are out of luck.  */
  assert_perror (err);
}
