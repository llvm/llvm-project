/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <string.h>

int
__getgroups (int n, gid_t *gidset)
{
  error_t err;
  int ngids;
  void *crit;

  if (n < 0)
    return __hurd_fail (EINVAL);

retry:
  crit = _hurd_critical_section_lock ();
  __mutex_lock (&_hurd_id.lock);

  if (err = _hurd_check_ids ())
    {
      __mutex_unlock (&_hurd_id.lock);
      _hurd_critical_section_unlock (crit);
      if (err == EINTR)
	/* Got a signal while inside an RPC of the critical section, retry again */
	goto retry;
      return __hurd_fail (err);
    }

  ngids = _hurd_id.gen.ngids;

  if (n != 0)
    {
      /* Copy the gids onto stack storage and then release the idlock.  */
      gid_t gids[ngids];
      memcpy (gids, _hurd_id.gen.gids, sizeof (gids));
      __mutex_unlock (&_hurd_id.lock);
      _hurd_critical_section_unlock (crit);

      /* Now that the lock is released, we can safely copy the
	 group set into the user's array, which might fault.  */
      if (ngids > n)
	return __hurd_fail (EINVAL);
      memcpy (gidset, gids, ngids * sizeof (gid_t));
    }
  else
    {
      __mutex_unlock (&_hurd_id.lock);
      _hurd_critical_section_unlock (crit);
    }

  return ngids;
}

weak_alias (__getgroups, getgroups)
