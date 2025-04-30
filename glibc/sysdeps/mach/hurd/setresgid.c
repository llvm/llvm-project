/* setresgid -- set real group ID, effective group ID, and saved-set group ID
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <hurd.h>
#include <hurd/id.h>

/* Set the real group ID, effective group ID, and saved-set group ID,
   of the calling process to RGID, EGID, and SGID, respectively.  */
int
__setresgid (gid_t rgid, gid_t egid, gid_t sgid)
{
  auth_t newauth;
  error_t err;

retry:
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);
  err = _hurd_check_ids ();

  if (!err)
    {
      /* Make a new auth handle which has EGID as the first element in the
	 list of effective gids.  */

      uid_t *newgen, *newaux;
      uid_t auxs[2] = { rgid, sgid };
      size_t ngen, naux;

      newgen = _hurd_id.gen.gids;
      ngen = _hurd_id.gen.ngids;
      if (egid != -1)
	{
	  if (_hurd_id.gen.ngids == 0)
	    {
	      /* No effective gids now.  The new set will be just UID.  */
	      newgen = &egid;
	      ngen = 1;
	    }
	  else
	    {
	      _hurd_id.gen.gids[0] = egid;
	      _hurd_id.valid = 0;
	    }
	}

      newaux = _hurd_id.aux.gids;
      naux = _hurd_id.aux.ngids;
      if (rgid != -1)
	{
	  if (_hurd_id.aux.ngids == 0)
	    {
	      newaux = &rgid;
	      naux = 1;
	    }
	  else
	    {
	      _hurd_id.aux.gids[0] = rgid;
	      _hurd_id.valid = 0;
	    }
	}

      if (sgid != -1)
	{
	  if (rgid == -1)
	    {
	      if (_hurd_id.aux.ngids >= 1)
		auxs[0] = _hurd_id.aux.gids[0];
	      else if (_hurd_id.gen.ngids >= 1)
		auxs[0] = _hurd_id.gen.gids[0];
	      else
		/* Not even an effective GID.
                   Fall back to the only GID we have. */
		auxs[0] = sgid;
	    }
	  if (_hurd_id.aux.ngids <= 1)
	    {
	      /* No saved gids now.  The new set will be just UID.  */
	      newaux = auxs;
	      naux = 2;
	    }
	  else
	    {
	      _hurd_id.aux.gids[1] = sgid;
	      _hurd_id.valid = 0;
	    }
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
libc_hidden_def (__setresgid)
weak_alias (__setresgid, setresgid)
