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
#include <hurd/port.h>
#include <hurd/id.h>
#include <hurdlock.h>
#include "set-hooks.h"

/* Things in the library which want to be run when the auth port changes.  */
DEFINE_HOOK (_hurd_reauth_hook, (auth_t new_auth));

static unsigned int reauth_lock = LLL_LOCK_INITIALIZER;

/* Set the auth port to NEW, and reauthenticate
   everything used by the library.  */
error_t
_hurd_setauth (auth_t new)
{
  error_t err;
  unsigned int d;
  mach_port_t newport, ref;

  /* Give the new send right a user reference.
     This is a good way to check that it is valid.  */
  if (err = __mach_port_mod_refs (__mach_task_self (), new,
				  MACH_PORT_RIGHT_SEND, 1))
    return err;

  HURD_CRITICAL_BEGIN;

  /* We lock against another thread doing setauth.  Anyone who sets
     _hurd_ports[INIT_PORT_AUTH] some other way is asking to lose.  */
  __mutex_lock (&reauth_lock);

  /* Install the new port in the cell.  */
  __mutex_lock (&_hurd_id.lock);
  _hurd_port_set (&_hurd_ports[INIT_PORT_AUTH], new);
  _hurd_id.valid = 0;
  if (_hurd_id.rid_auth)
    {
      __mach_port_deallocate (__mach_task_self (), _hurd_id.rid_auth);
      _hurd_id.rid_auth = MACH_PORT_NULL;
    }
  __mutex_unlock (&_hurd_id.lock);

  if (_hurd_init_dtable != NULL)
    /* We just have the simple table we got at startup.
       Otherwise, a reauth_hook in dtable.c takes care of this.  */
    for (d = 0; d < _hurd_init_dtablesize; ++d)
      if (_hurd_init_dtable[d] != MACH_PORT_NULL)
	{
	  mach_port_t new;
	  ref = __mach_reply_port ();
	  if (! __io_reauthenticate (_hurd_init_dtable[d],
				     ref, MACH_MSG_TYPE_MAKE_SEND)
	      && ! HURD_PORT_USE (&_hurd_ports[INIT_PORT_AUTH],
				  __auth_user_authenticate
				  (port,
				   ref, MACH_MSG_TYPE_MAKE_SEND,
				   &new)))
	    {
	      __mach_port_deallocate (__mach_task_self (),
				      _hurd_init_dtable[d]);
	      _hurd_init_dtable[d] = new;
	    }
	  __mach_port_destroy (__mach_task_self (), ref);
	}

  ref = __mach_reply_port ();
  if (__USEPORT (CRDIR,
		 ! __io_reauthenticate (port,
					ref, MACH_MSG_TYPE_MAKE_SEND)
		 && ! __auth_user_authenticate (new,
						ref, MACH_MSG_TYPE_MAKE_SEND,
						&newport)))
    _hurd_port_set (&_hurd_ports[INIT_PORT_CRDIR], newport);
  __mach_port_destroy (__mach_task_self (), ref);

  ref = __mach_reply_port ();
  if (__USEPORT (CWDIR,
		 ! __io_reauthenticate (port,
					ref, MACH_MSG_TYPE_MAKE_SEND)
		 && ! __auth_user_authenticate (new,
						ref, MACH_MSG_TYPE_MAKE_SEND,
						&newport)))
    _hurd_port_set (&_hurd_ports[INIT_PORT_CWDIR], newport);
  __mach_port_destroy (__mach_task_self (), ref);

  /* Run things which want to do reauthorization stuff.  */
  RUN_HOOK (_hurd_reauth_hook, (new));

  __mutex_unlock (&reauth_lock);

  HURD_CRITICAL_END;

  return 0;
}

int
__setauth (auth_t new)
{
  error_t err = _hurd_setauth (new);
  return err ? __hurd_fail (err) : 0;
}

weak_alias (__setauth, setauth)
