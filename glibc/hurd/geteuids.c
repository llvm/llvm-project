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

#include <hurd.h>
#include <hurd/id.h>
#include <string.h>

int
geteuids (int n, uid_t *uidset)
{
  error_t err;
  int nuids;
  void *crit;

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

  nuids = _hurd_id.gen.nuids;

  if (n != 0)
    {
      /* Copy the uids onto stack storage and then release the idlock.  */
      uid_t uids[nuids];
      memcpy (uids, _hurd_id.gen.uids, sizeof (uids));
      __mutex_unlock (&_hurd_id.lock);
      _hurd_critical_section_unlock (crit);

      /* Now that the lock is released, we can safely copy the
	 uid set into the user's array, which might fault.  */
      if (nuids > n)
	nuids = n;
      memcpy (uidset, uids, nuids * sizeof (uid_t));
    }
  else
    {
      __mutex_unlock (&_hurd_id.lock);
      _hurd_critical_section_unlock (crit);
    }

  return nuids;
}

/* XXX Remove this alias when we bump the libc soname.  */

#ifdef SHARED
weak_alias (geteuids, __getuids)
#endif
