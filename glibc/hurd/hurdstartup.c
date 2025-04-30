/* Initial program startup for running under the GNU Hurd.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <hurd.h>
#include <hurd/exec_startup.h>
#include <sysdep.h>
#include <unistd.h>
#include <elf.h>
#include <set-hooks.h>
#include "hurdstartup.h"
#include <argz.h>

mach_port_t *_hurd_init_dtable;
mach_msg_type_number_t _hurd_init_dtablesize;

extern void __mach_init (void);

/* Entry point.  This is the first thing in the text segment.

   The exec server started the initial thread in our task with this spot the
   PC, and a stack that is presumably big enough.  We do basic Mach
   initialization so mig-generated stubs work, and then do an exec_startup
   RPC on our bootstrap port, to which the exec server responds with the
   information passed in the exec call, as well as our original bootstrap
   port, and the base address and size of the preallocated stack.  */


void
_hurd_startup (void **argptr, void (*main) (intptr_t *data))
{
  error_t err;
  mach_port_t in_bootstrap;
  char *args, *env;
  mach_msg_type_number_t argslen, envlen;
  struct hurd_startup_data data;
  char **argv, **envp;
  int argc, envc;
  intptr_t *argcptr;
  vm_address_t addr;

  /* Attempt to map page zero redzoned before we receive any RPC
     data that might get allocated there.  We can ignore errors.  */
  addr = 0;
  __vm_map (__mach_task_self (),
	    &addr, __vm_page_size, 0, 0, MACH_PORT_NULL, 0, 1,
	    VM_PROT_NONE, VM_PROT_NONE, VM_INHERIT_COPY);

  if (err = __task_get_special_port (__mach_task_self (), TASK_BOOTSTRAP_PORT,
				     &in_bootstrap))
    LOSE;

  if (in_bootstrap != MACH_PORT_NULL)
    {
      /* Call the exec server on our bootstrap port and
	 get all our standard information from it.  */

      argslen = envlen = 0;
      data.dtablesize = data.portarraysize = data.intarraysize = 0;

      err = __exec_startup_get_info (in_bootstrap,
				     &data.user_entry,
				     &data.phdr, &data.phdrsz,
				     &data.stack_base, &data.stack_size,
				     &data.flags,
				     &args, &argslen,
				     &env, &envlen,
				     &data.dtable, &data.dtablesize,
				     &data.portarray, &data.portarraysize,
				     &data.intarray, &data.intarraysize);
      __mach_port_deallocate (__mach_task_self (), in_bootstrap);
    }

  if (err || in_bootstrap == MACH_PORT_NULL || (data.flags & EXEC_STACK_ARGS))
    {
      /* Either we have no bootstrap port, or the RPC to the exec server
	 failed, or whoever started us up passed the flag saying args are
	 on the stack.  Try to snarf the args in the canonical Mach way.
	 Hopefully either they will be on the stack as expected, or the
	 stack will be zeros so we don't crash.  */

      argcptr = (intptr_t *) argptr;
      argc = argcptr[0];
      argv = (char **) &argcptr[1];
      envp = &argv[argc + 1];
      envc = 0;
      while (envp[envc])
	++envc;
    }
  else
    {
      /* Turn the block of null-separated strings we were passed for the
	 arguments and environment into vectors of pointers to strings.  */

      /* Count up the arguments so we can allocate ARGV.  */
      argc = __argz_count (args, argslen);
      /* Count up the environment variables so we can allocate ENVP.  */
      envc = __argz_count (env, envlen);

      /* There were some arguments.  Allocate space for the vectors of
	 pointers and fill them in.  We allocate the space for the
	 environment pointers immediately after the argv pointers because
	 the ELF ABI will expect it.  */
      argcptr = __alloca (sizeof (intptr_t)
			  + (argc + 1 + envc + 1) * sizeof (char *)
			  + sizeof (struct hurd_startup_data));
      *argcptr = argc;
      argv = (void *) (argcptr + 1);
      __argz_extract (args, argslen, argv);

      /* There was some environment.  */
      envp = &argv[argc + 1];
      __argz_extract (env, envlen, envp);
    }

  if (err || in_bootstrap == MACH_PORT_NULL)
    {
      /* Either we have no bootstrap port, or the RPC to the exec server
	 failed.  Set all our other variables to have empty information.  */

      data.flags = 0;
      args = env = NULL;
      argslen = envlen = 0;
      data.dtable = NULL;
      data.dtablesize = 0;
      data.portarray = NULL;
      data.portarraysize = 0;
      data.intarray = NULL;
      data.intarraysize = 0;
    }
  else if ((void *) &envp[envc + 1] == argv[0])
    {
      /* The arguments arrived on the stack from the kernel, but our
	 protocol requires some space after them for a `struct
	 hurd_startup_data'.  Move them.  */
      struct
	{
	  intptr_t count;
	  char *argv[argc + 1];
	  char *envp[envc + 1];
	  struct hurd_startup_data data;
	} *args = alloca (sizeof *args);
      if ((void *) &args[1] == (void *) argcptr)
	args = alloca (-((char *) &args->data - (char *) args));
      memmove (args, argcptr, (char *) &args->data - (char *) args);
      argcptr = (void *) args;
      argv = args->argv;
      envp = args->envp;
    }

  {
    struct hurd_startup_data *d = (void *) &envp[envc + 1];

    if ((void *) d != argv[0])
      {
	*d = data;
	_hurd_init_dtable = d->dtable;
	_hurd_init_dtablesize = d->dtablesize;
      }

    (*main) (argcptr);
  }

  /* Should never get here.  */
  LOSE;
  abort ();
}
