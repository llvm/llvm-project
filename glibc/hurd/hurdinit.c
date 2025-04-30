/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/port.h>
#include <ldsodefs.h>
#include "set-hooks.h"
#include "hurdmalloc.h"		/* XXX */


int _hurd_exec_flags;
struct hurd_port *_hurd_ports;
unsigned int _hurd_nports;
mode_t _hurd_umask;
sigset_t _hurdsig_traced;

char **__libc_argv;
int __libc_argc;

static int *_hurd_intarray;
static size_t _hurd_intarraysize;
static mach_port_t *_hurd_portarray;
static size_t _hurd_portarraysize;

error_t
_hurd_ports_use (int which, error_t (*operate) (mach_port_t))
{
  if (__glibc_unlikely (_hurd_ports == NULL))
    /* This means that _hurd_init has not been called yet, which is
       normally only the case in the bootstrap filesystem, and there
       only in the early phases of booting.  */
    return EGRATUITOUS;

  return HURD_PORT_USE (&_hurd_ports[which], (*operate) (port));
}

DEFINE_HOOK (_hurd_subinit, (void));

__typeof (_hurd_proc_init) _hurd_new_proc_init;	/* below */

/* Initialize the library data structures from the
   ints and ports passed to us by the exec server.

   PORTARRAY and INTARRAY are vm_deallocate'd.  */

void
_hurd_init (int flags, char **argv,
	    mach_port_t *portarray, size_t portarraysize,
	    int *intarray, size_t intarraysize)
{
  size_t i;

  _hurd_exec_flags = flags;

  _hurd_ports = malloc (portarraysize * sizeof (*_hurd_ports));
  if (_hurd_ports == NULL)
    __libc_fatal ("Can't allocate _hurd_ports\n");
  _hurd_nports = portarraysize;

  /* See what ports we were passed.  */
  for (i = 0; i < portarraysize; ++i)
    _hurd_port_init (&_hurd_ports[i], portarray[i]);

  /* When the user asks for the bootstrap port,
     he will get the one the exec server passed us.  */
  __task_set_special_port (__mach_task_self (), TASK_BOOTSTRAP_PORT,
			   portarray[INIT_PORT_BOOTSTRAP]);

  if (intarraysize > INIT_UMASK)
    _hurd_umask = intarray[INIT_UMASK] & 0777;
  else
    _hurd_umask = CMASK;

  if (intarraysize > INIT_TRACEMASK)
    _hurdsig_traced = intarray[INIT_TRACEMASK];

  _hurd_intarray = intarray;
  _hurd_intarraysize = intarraysize;
  _hurd_portarray = portarray;
  _hurd_portarraysize = portarraysize;

  if (flags & EXEC_SECURE)
    {
      /* XXX if secure exec, elide environment variables
	 which the library uses and could be security holes.
	 CORESERVER, COREFILE
      */
    }

  /* Call other things which want to do some initialization.  These are not
     on the __libc_subinit hook because things there like to be able to
     assume the availability of the POSIX.1 services we provide.  */
  RUN_HOOK (_hurd_subinit, ());
}
libc_hidden_def (_hurd_init)

void
_hurd_libc_proc_init (char **argv)
{
  if (_hurd_portarray)
    {
      /* We will start the signal thread, so we need to initialize libpthread
       * if linked in.  */
      if (__pthread_initialize_minimal != NULL)
	__pthread_initialize_minimal ();

      /* Tell the proc server we exist, if it does.  */
      if (_hurd_portarray[INIT_PORT_PROC] != MACH_PORT_NULL)
	_hurd_new_proc_init (argv, _hurd_intarray, _hurd_intarraysize);

      /* All done with init ints and ports.  */
      __vm_deallocate (__mach_task_self (),
		       (vm_address_t) _hurd_intarray,
		       _hurd_intarraysize * sizeof (int));
      _hurd_intarray = NULL;
      _hurd_intarraysize = 0;

      __vm_deallocate (__mach_task_self (),
		       (vm_address_t) _hurd_portarray,
		       _hurd_portarraysize * sizeof (mach_port_t));
      _hurd_portarray = NULL;
      _hurd_portarraysize = 0;
    }
}
libc_hidden_def (_hurd_libc_proc_init)

#include <hurd/signal.h>

/* The user can do "int _hide_arguments = 1;" to make
   sure the arguments are never visible with `ps'.  */
int _hide_arguments, _hide_environment;

/* Hook for things which should be initialized as soon as the proc
   server is available.  */
DEFINE_HOOK (_hurd_proc_subinit, (void));

/* Do startup handshaking with the proc server just installed in _hurd_ports.
   Call _hurdsig_init to set up signal processing.  */

void
_hurd_new_proc_init (char **argv,
		     const int *intarray, size_t intarraysize)
{
  mach_port_t oldmsg;
  struct hurd_userlink ulink;
  process_t procserver;

  /* Initialize the signal code; Mach exceptions will become signals.  */
  _hurdsig_init (intarray, intarraysize);

  /* The signal thread is now prepared to receive messages.
     It is safe to give the port to the proc server.  */

  procserver = _hurd_port_get (&_hurd_ports[INIT_PORT_PROC], &ulink);

  /* Give the proc server our message port.  */
  __proc_setmsgport (procserver, _hurd_msgport, &oldmsg);
  if (oldmsg != MACH_PORT_NULL)
    /* Deallocate the old msg port we replaced.  */
    __mach_port_deallocate (__mach_task_self (), oldmsg);

  /* Tell the proc server where our args and environment are.  */
  __proc_set_arg_locations (procserver,
			    _hide_arguments ? 0 : (vm_address_t) argv,
			    _hide_environment ? 0 : (vm_address_t) __environ);

  _hurd_port_free (&_hurd_ports[INIT_PORT_PROC], &ulink, procserver);

  /* Initialize proc server-assisted fault recovery for the signal thread.  */
  _hurdsig_fault_init ();

  /* Call other things which want to do some initialization.  These are not
     on the _hurd_subinit hook because things there assume that things done
     here, like _hurd_pid, are already initialized.  */
  RUN_HOOK (_hurd_proc_subinit, ());

  /* XXX This code should probably be removed entirely at some point.  This
     conditional should make it reasonably usable with old gdb's for a
     while.  Eventually it probably makes most sense for the exec server to
     mask out EXEC_SIGTRAP so the debugged program is closer to not being
     able to tell it's being debugged.  */
  if (!__sigisemptyset (&_hurdsig_traced)
#ifdef EXEC_SIGTRAP
      && !(_hurd_exec_flags & EXEC_SIGTRAP)
#endif
      )
    /* This process is "traced", meaning it should stop on signals or exec.
       We are all set up now to handle signals.  Stop ourselves, to inform
       our parent (presumably a debugger) that the exec has completed.  */
    __msg_sig_post (_hurd_msgport, SIGTRAP, TRAP_TRACE, __mach_task_self ());
}

#include <shlib-compat.h>
versioned_symbol (libc, _hurd_new_proc_init, _hurd_proc_init, GLIBC_2_1);

/* Called when we get a message telling us to change our proc server port.  */

error_t
_hurd_setproc (process_t procserver)
{
  error_t err;
  mach_port_t oldmsg;

  /* Give the proc server our message port.  */
  if (err = __proc_setmsgport (procserver, _hurd_msgport, &oldmsg))
    return err;
  if (oldmsg != MACH_PORT_NULL)
    /* Deallocate the old msg port we replaced.  */
    __mach_port_deallocate (__mach_task_self (), oldmsg);

  /* Tell the proc server where our args and environment are.  */
  if (err = __proc_set_arg_locations (procserver,
				      _hide_arguments ? 0
				      : (vm_address_t) __libc_argv,
				      _hide_environment ? 0
				      : (vm_address_t) __environ))
    return err;

  /* Those calls worked, so the port looks good.  */
  _hurd_port_set (&_hurd_ports[INIT_PORT_PROC], procserver);

  {
    pid_t oldpgrp = _hurd_pgrp;

    /* Call these functions again so they can fetch the
       new information from the new proc server.  */
    RUN_HOOK (_hurd_proc_subinit, ());

    if (_hurd_pgrp != oldpgrp)
      {
	/* Run things that want notification of a pgrp change.  */
	DECLARE_HOOK (_hurd_pgrp_changed_hook, (pid_t));
	RUN_HOOK (_hurd_pgrp_changed_hook, (_hurd_pgrp));
      }
  }

  return 0;
}
