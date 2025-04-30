/* Start thread.  Mach version.
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

#include <pt-internal.h>

/* Start THREAD.  Get the kernel thread scheduled and running.  */
int
__pthread_thread_start (struct __pthread *thread)
{
  static int do_start;
  error_t err;

  if (!do_start)
    {
      /* The main thread is already running: do nothing.  */
      assert (__pthread_total == 1);
      assert ((
		{
		  mach_port_t ktid = __mach_thread_self ();
		  int ok = thread->kernel_thread == ktid;
		  __mach_port_deallocate (__mach_task_self (),
					  thread->kernel_thread);
		  ok;
		}));
      do_start = 1;
    }
  else
    {
      err = __thread_resume (thread->kernel_thread);
      assert_perror (err);
    }

  return 0;
}
