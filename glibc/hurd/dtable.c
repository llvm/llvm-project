/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <hurd/term.h>
#include <hurd/fd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <limits.h>
#include <lock-intern.h>	/* For `struct mutex'.  */
#include "set-hooks.h"
#include "hurdmalloc.h"		/* XXX */


struct mutex _hurd_dtable_lock = MUTEX_INITIALIZER; /* XXX ld bug; must init */
struct hurd_fd **_hurd_dtable;
int _hurd_dtablesize;


DEFINE_HOOK (_hurd_fd_subinit, (void));

/* Initialize the file descriptor table at startup.  */

static void
init_dtable (void)
{
  int i;

  __mutex_init (&_hurd_dtable_lock);

  /* The initial size of the descriptor table is that of the passed-in
     table.  It will be expanded as necessary up to _hurd_dtable_rlimit.  */
  _hurd_dtablesize = _hurd_init_dtablesize;

  /* Allocate the vector of pointers.  */
  _hurd_dtable = malloc (_hurd_dtablesize * sizeof (*_hurd_dtable));
  if (_hurd_dtablesize != 0 && _hurd_dtable == NULL)
    __libc_fatal ("hurd: Can't allocate file descriptor table\n");

  /* Initialize the descriptor table.  */
  for (i = 0; (unsigned int) i < _hurd_init_dtablesize; ++i)
    {
      if (_hurd_init_dtable[i] == MACH_PORT_NULL)
	/* An unused descriptor is marked by a null pointer.  */
	_hurd_dtable[i] = NULL;
      else
	{
	  /* Allocate a new file descriptor structure.  */
	  struct hurd_fd *new = malloc (sizeof (struct hurd_fd));
	  if (new == NULL)
	    __libc_fatal ("hurd: Can't allocate initial file descriptors\n");

	  /* Initialize the port cells.  */
	  _hurd_port_init (&new->port, MACH_PORT_NULL);
	  _hurd_port_init (&new->ctty, MACH_PORT_NULL);

	  /* Install the port in the descriptor.
	     This sets up all the ctty magic.  */
	  _hurd_port2fd (new, _hurd_init_dtable[i], 0);

	  _hurd_dtable[i] = new;
	}
    }

  /* Clear out the initial descriptor table.
     Everything must use _hurd_dtable now.  */
  __vm_deallocate (__mach_task_self (),
		   (vm_address_t) _hurd_init_dtable,
		   _hurd_init_dtablesize * sizeof (_hurd_init_dtable[0]));
  _hurd_init_dtable = NULL;
  _hurd_init_dtablesize = 0;

  /* Initialize the remaining empty slots in the table.  */
  for (; i < _hurd_dtablesize; ++i)
    _hurd_dtable[i] = NULL;

  /* Run things that want to run after the file descriptor table
     is initialized.  */
  RUN_HOOK (_hurd_fd_subinit, ());

  (void) &init_dtable;		/* Avoid "defined but not used" warning.  */
}

text_set_element (_hurd_subinit, init_dtable);

/* XXX when the linker supports it, the following functions should all be
   elsewhere and just have text_set_elements here.  */

/* Called by `getdport' to do its work.  */

static file_t
get_dtable_port (int fd)
{
  struct hurd_fd *d = _hurd_fd_get (fd);
  file_t dport;

  if (!d)
    return __hurd_fail (EBADF), MACH_PORT_NULL;

  HURD_CRITICAL_BEGIN;

  dport = HURD_PORT_USE (&d->port,
			 ({
			   error_t err;
			   mach_port_t outport;
			   err = __mach_port_mod_refs (__mach_task_self (),
						       port,
						       MACH_PORT_RIGHT_SEND,
						       1);
			   if (err)
			     {
			       errno = err;
			       outport = MACH_PORT_NULL;
			     }
			   else
			     outport = port;
			   outport;
			 }));

  HURD_CRITICAL_END;

  return dport;
}

file_t (*_hurd_getdport_fn) (int fd) = get_dtable_port;

#include <hurd/signal.h>

/* We are in the child fork; the dtable lock is still held.
   The parent has inserted send rights for all the normal io ports,
   but we must recover ctty-special ports for ourselves.  */
static error_t
fork_child_dtable (void)
{
  error_t err;
  int i;

  err = 0;

  for (i = 0; !err && i < _hurd_dtablesize; ++i)
    {
      struct hurd_fd *d = _hurd_dtable[i];
      if (d == NULL)
	continue;

      /* No other thread is using the send rights in the child task.  */
      d->port.users = d->ctty.users = NULL;

      if (d->ctty.port != MACH_PORT_NULL)
	{
	  /* There was a ctty-special port in the parent.
	     We need to get one for ourselves too.  */
	  __mach_port_deallocate (__mach_task_self (), d->ctty.port);
	  err = __term_open_ctty (d->port.port, _hurd_pid, _hurd_pgrp,
				  &d->ctty.port);
	  if (err)
	    d->ctty.port = MACH_PORT_NULL;
	}

      /* XXX for each fd with a cntlmap, reauth and re-map_cntl.  */
    }
  return err;

  (void) &fork_child_dtable;	/* Avoid "defined but not used" warning.  */
}

data_set_element (_hurd_fork_locks, _hurd_dtable_lock);	/* XXX ld bug: bss */
text_set_element (_hurd_fork_child_hook, fork_child_dtable);

/* Called when our process group has changed.  */

static void
ctty_new_pgrp (void)
{
  int i;

retry:
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_dtable_lock);

  if (__USEPORT (CTTYID, port == MACH_PORT_NULL))
    {
      /* We have no controlling terminal.  If we haven't had one recently,
	 but our pgrp is being pointlessly diddled anyway, then we will
	 have nothing to do in the loop below because no fd will have a
	 ctty port at all.

	 More likely, a setsid call is responsible both for the change
	 in pgrp and for clearing the cttyid port.  In that case, setsid
	 held the dtable lock while updating the dtable to clear all the
	 ctty ports, and ergo must have finished doing so before we run here.
	 So we can be sure, again, that the loop below has no work to do.  */
    }
  else
    for (i = 0; i < _hurd_dtablesize; ++i)
      {
	struct hurd_fd *const d = _hurd_dtable[i];
	struct hurd_userlink ulink, ctty_ulink;
	io_t port, ctty;

	if (d == NULL)
	  /* Nothing to do for an unused descriptor cell.  */
	  continue;

	port = _hurd_port_get (&d->port, &ulink);
	ctty = _hurd_port_get (&d->ctty, &ctty_ulink);

	if (ctty != MACH_PORT_NULL)
	  {
	    /* This fd has a ctty-special port.  We need a new one, to tell
	       the io server of our different process group.  */
	    io_t new;
	    error_t err;
	    if ((err = __term_open_ctty (port, _hurd_pid, _hurd_pgrp, &new)))
	      {
		if (err == EINTR)
		  {
		    /* Got a signal while inside an RPC of the critical section, retry again */
		    __mutex_unlock (&_hurd_dtable_lock);
		    HURD_CRITICAL_UNLOCK;
		    goto retry;
		  }
		new = MACH_PORT_NULL;
	      }
	    _hurd_port_set (&d->ctty, new);
	  }

	_hurd_port_free (&d->port, &ulink, port);
	_hurd_port_free (&d->ctty, &ctty_ulink, ctty);
      }

  __mutex_unlock (&_hurd_dtable_lock);
  HURD_CRITICAL_END;

  (void) &ctty_new_pgrp;	/* Avoid "defined but not used" warning.  */
}

text_set_element (_hurd_pgrp_changed_hook, ctty_new_pgrp);

/* Called to reauthenticate the dtable when the auth port changes.  */

static void
reauth_dtable (void)
{
  int i;

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_dtable_lock);

  for (i = 0; i < _hurd_dtablesize; ++i)
    {
      struct hurd_fd *const d = _hurd_dtable[i];
      mach_port_t new, newctty, ref;

      if (d == NULL)
	/* Nothing to do for an unused descriptor cell.  */
	continue;

      ref = __mach_reply_port ();

      /* Take the descriptor cell's lock.  */
      __spin_lock (&d->port.lock);

      /* Reauthenticate the descriptor's port.  */
      if (d->port.port != MACH_PORT_NULL
	  && ! __io_reauthenticate (d->port.port,
				    ref, MACH_MSG_TYPE_MAKE_SEND)
	  && ! __USEPORT (AUTH, __auth_user_authenticate
			  (port,
			   ref, MACH_MSG_TYPE_MAKE_SEND,
			   &new)))
	{
	  /* Replace the port in the descriptor cell
	     with the newly reauthenticated port.  */

	  if (d->ctty.port != MACH_PORT_NULL
	      && ! __io_reauthenticate (d->ctty.port,
					ref, MACH_MSG_TYPE_MAKE_SEND)
	      && ! __USEPORT (AUTH, __auth_user_authenticate
			      (port,
			       ref, MACH_MSG_TYPE_MAKE_SEND,
			       &newctty)))
	    _hurd_port_set (&d->ctty, newctty);

	  _hurd_port_locked_set (&d->port, new);
	}
      else
	/* Lost.  Leave this descriptor cell alone.  */
	__spin_unlock (&d->port.lock);

      __mach_port_destroy (__mach_task_self (), ref);
    }

  __mutex_unlock (&_hurd_dtable_lock);
  HURD_CRITICAL_END;

  (void) &reauth_dtable;	/* Avoid "defined but not used" warning.  */
}

text_set_element (_hurd_reauth_hook, reauth_dtable);
