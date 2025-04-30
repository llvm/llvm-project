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

/* Set the group ID of the calling process to UID.
   If the calling process is the super-user, the real
   and effective group IDs, and the saved set-group-ID to UID;
   if not, the effective group ID is set to GID.  */
int
__setgid (gid_t gid)
{
  auth_t newauth;
  error_t err;

retry:
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);
  err = _hurd_check_ids ();

  if (!err)
    {
      /* Make a new auth handle which has GID as the real gid,
	 and as the first element in the list of effective gids.  */

      gid_t *newgen, *newaux, auxbuf[2];
      size_t ngen, naux;

      if (_hurd_id.gen.ngids == 0)
	{
	  /* No effective gids now.  The new set will be just GID.  */
	  newgen = &gid;
	  ngen = 1;
	}
      else
	{
	  _hurd_id.gen.gids[0] = gid;
	  _hurd_id.valid = 0;
	  newgen = _hurd_id.gen.gids;
	  ngen = _hurd_id.gen.ngids;
	}

      newaux = _hurd_id.aux.gids;
      naux = _hurd_id.aux.ngids;
      if (_hurd_id.gen.nuids > 0 && _hurd_id.gen.uids[0] == 0)
	{
	  /* We are root; set the real and saved IDs too.  */
	  _hurd_id.valid = 0;
	  if (_hurd_id.aux.ngids < 2)
	    {
	      newaux = auxbuf;
	      naux = 2;
	    }
	  newaux[0] = newaux[1] = gid;
	}

      err = __USEPORT (AUTH, __auth_makeauth
		       (port, NULL, MACH_MSG_TYPE_COPY_SEND, 0,
			_hurd_id.gen.uids, _hurd_id.gen.nuids,
			_hurd_id.aux.uids, _hurd_id.aux.nuids,
			newgen, ngen, newaux, naux,
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

weak_alias (__setgid, setgid)
