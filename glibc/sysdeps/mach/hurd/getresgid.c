/* getresgid -- fetch real group ID, effective group ID, and saved-set group ID
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

/* Fetch the real group ID, effective group ID, and saved-set group ID,
   of the calling process.  */
int
__getresgid (gid_t *rgid, gid_t *egid, gid_t *sgid)
{
  error_t err;

retry:
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);

  err = _hurd_check_ids ();
  if (!err)
    {
      if (_hurd_id.aux.ngids < 1)
	/* We do not even have a real GID.  */
	err = EGRATUITOUS;
      else
	{
	  gid_t real = _hurd_id.aux.gids[0];

	  *rgid = real;
	  *egid = _hurd_id.gen.ngids < 1 ? real : _hurd_id.gen.gids[0];
	  *sgid = _hurd_id.aux.ngids < 2 ? real : _hurd_id.aux.gids[1];
	}
    }

  __mutex_unlock (&_hurd_id.lock);
  HURD_CRITICAL_END;
  if (err == EINTR)
    /* Got a signal while inside an RPC of the critical section, retry again */
    goto retry;

  return __hurd_fail (err);
}
libc_hidden_def (__getresgid)
weak_alias (__getresgid, getresgid)
