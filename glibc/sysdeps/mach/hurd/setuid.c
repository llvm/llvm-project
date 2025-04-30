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

#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <hurd.h>
#include <hurd/id.h>
#include <string.h>

/* Set the user ID of the calling process to UID.
   If the calling process is the super-user, the real
   and effective user IDs, and the saved set-user-ID to UID;
   if not, the effective user ID is set to UID.  */
int
__setuid (uid_t uid)
{
  auth_t newauth;
  error_t err;

retry:
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);
  err = _hurd_check_ids ();

  if (!err)
    {
      /* Make a new auth handle which has UID as the real uid,
	 and as the first element in the list of effective uids.  */

      uid_t *newgen, *newaux, auxbuf[2];
      size_t ngen, naux;

      newaux = _hurd_id.aux.uids;
      naux = _hurd_id.aux.nuids;
      if (_hurd_id.gen.nuids == 0)
	{
	  /* No effective uids now.  The new set will be just UID.  */
	  newgen = &uid;
	  ngen = 1;
	}
      else if (_hurd_id.gen.uids[0] == 0)
	{
	  /* We are root; set the effective, real, and saved to UID.  */
	  _hurd_id.gen.uids[0] = uid;
	  _hurd_id.valid = 0;
	  newgen = _hurd_id.gen.uids;
	  ngen = _hurd_id.gen.nuids;
	  if (_hurd_id.aux.nuids < 2)
	    {
	      newaux = auxbuf;
	      naux = 2;
	    }
	  newaux[0] = newaux[1] = uid;
	}
      else
	{
	  /* We are not root; just change the effective UID.  */
	  /* XXX that implies an unprivileged setuid(0) will give
	     the caller root, no questions asked! */
	  _hurd_id.gen.uids[0] = uid;
	  _hurd_id.valid = 0;
	  newgen = _hurd_id.gen.uids;
	  ngen = _hurd_id.gen.nuids;
	}

      err = __USEPORT (AUTH, __auth_makeauth
		       (port, NULL, MACH_MSG_TYPE_COPY_SEND, 0,
			newgen, ngen, newaux, naux,
			_hurd_id.gen.gids, _hurd_id.gen.ngids,
			_hurd_id.aux.gids, _hurd_id.aux.ngids,
			&newauth));
    }
  __mutex_unlock (&_hurd_id.lock);
  HURD_CRITICAL_END;
  if (err == EINTR)
    /* Got a signal while inside an RPC of the critical section, retry again */
    goto retry;

  if (err)
    return __hurd_fail (err);

  /* Install the new handle and reauthenticate everything.  */
  err = __setauth (newauth);
  __mach_port_deallocate (__mach_task_self (), newauth);
  return err;
}

weak_alias (__setuid, setuid)
