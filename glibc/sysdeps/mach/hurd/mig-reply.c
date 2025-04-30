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

#include <mach.h>
#include <mach/mig_support.h>
#include <hurd/threadvar.h>

/* These functions are called by MiG-generated code.  */

mach_port_t __hurd_reply_port0;

/* Called by MiG to get a reply port.  */
mach_port_t
__mig_get_reply_port (void)
{
  if (__hurd_local_reply_port == MACH_PORT_NULL
      || (&__hurd_local_reply_port != &__hurd_reply_port0
	  && __hurd_local_reply_port == __hurd_reply_port0))
    __hurd_local_reply_port = __mach_reply_port ();

  return __hurd_local_reply_port;
}
weak_alias (__mig_get_reply_port, mig_get_reply_port)
libc_hidden_def (__mig_get_reply_port)

/* Called by MiG to deallocate the reply port.  */
void
__mig_dealloc_reply_port (mach_port_t arg)
{
  mach_port_t port = __hurd_local_reply_port;
  __hurd_local_reply_port = MACH_PORT_NULL;	/* So the mod_refs RPC won't use it.  */

  if (MACH_PORT_VALID (port))
    __mach_port_mod_refs (__mach_task_self (), port,
			  MACH_PORT_RIGHT_RECEIVE, -1);
}
weak_alias (__mig_dealloc_reply_port, mig_dealloc_reply_port)
libc_hidden_def (__mig_dealloc_reply_port)

/* Called by mig interfaces when done with a port.  Used to provide the
   same interface as needed when a custom allocator is used.  */
void
__mig_put_reply_port(mach_port_t port)
{
  /* Do nothing.  */
}
weak_alias (__mig_put_reply_port, mig_put_reply_port)

/* Called at startup with STACK == NULL.  When per-thread variables are set
   up, this is called again with STACK set to the new stack being switched
   to, where per-thread variables should be set up.  */
void
__mig_init (void *stack)
{
  /* Do nothing.  */
}
weak_alias (__mig_init, mig_init)
libc_hidden_def (__mig_init)
