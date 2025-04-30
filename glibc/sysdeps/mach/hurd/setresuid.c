/* setresuid -- set real user ID, effective user ID, and saved-set user ID
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

/* Set the real user ID, effective user ID, and saved-set user ID,
   of the calling process to RUID, EUID, and SUID, respectively.  */
int
__setresuid (uid_t ruid, uid_t euid, uid_t suid)
{
  auth_t newauth;
  error_t err;

retry:
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);
  err = _hurd_check_ids ();

  if (!err)
    {
      /* Make a new auth handle which has EUID as the first element in the
	 list of effective uids.  */

      uid_t *newgen, *newaux;
      uid_t auxs[2] = { ruid, suid };
      size_t ngen, naux;

      newgen = _hurd_id.gen.uids;
      ngen = _hurd_id.gen.nuids;
      if (euid != -1)
	{
	  if (_hurd_id.gen.nuids == 0)
	    {
	      /* No effective uids now.  The new set will be just UID.  */
	      newgen = &euid;
	      ngen = 1;
	    }
	  else
	    {
	      _hurd_id.gen.uids[0] = euid;
	      _hurd_id.valid = 0;
	    }
	}

      newaux = _hurd_id.aux.uids;
      naux = _hurd_id.aux.nuids;
      if (ruid != -1)
	{
	  if (_hurd_id.aux.nuids == 0)
	    {
	      newaux = &ruid;
	      naux = 1;
	    }
	  else
	    {
	      _hurd_id.aux.uids[0] = ruid;
	      _hurd_id.valid = 0;
	    }
	}

      if (suid != -1)
	{
	  if (ruid == -1)
	    {
	      if (_hurd_id.aux.nuids >= 1)
		auxs[0] = _hurd_id.aux.uids[0];
	      else if (_hurd_id.gen.nuids >= 1)
		auxs[0] = _hurd_id.gen.uids[0];
	      else
		/* Not even an effective UID.
                   Fall back to the only UID we have. */
		auxs[0] = suid;
	    }
	  if (_hurd_id.aux.nuids <= 1)
	    {
	      /* No saved uids now.  The new set will be just UID.  */
	      newaux = auxs;
	      naux = 2;
	    }
	  else
	    {
	      _hurd_id.aux.uids[1] = suid;
	      _hurd_id.valid = 0;
	    }
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
libc_hidden_def (__setresuid)
weak_alias (__setresuid, setresuid)
