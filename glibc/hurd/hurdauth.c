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
#include <hurd/msg_server.h>
#include <hurd/id.h>
#include <string.h>

int
_hurd_refport_secure_p (mach_port_t ref)
{
  if (ref == __mach_task_self ())
    return 1;
  if (__USEPORT (AUTH, ref == port))
    return 1;
  return 0;
}

kern_return_t
_S_msg_add_auth (mach_port_t me,
		 auth_t addauth)
{
  error_t err;
  auth_t newauth;
  uid_t *genuids, *gengids, *auxuids, *auxgids;
  mach_msg_type_number_t ngenuids, ngengids, nauxuids, nauxgids;
  uid_t *newgenuids, *newgengids, *newauxuids, *newauxgids;
  mach_msg_type_number_t nnewgenuids, nnewgengids, nnewauxuids, nnewauxgids;

  /* Create a list of ids and store it in NEWLISTP, length NEWLISTLEN.
     Keep all the ids in EXIST (len NEXIST), adding in those from NEW
     (len NNEW) which are not already there.  */
  error_t make_list (uid_t **newlistp, mach_msg_type_number_t *newlistlen,
		     uid_t *exist, mach_msg_type_number_t nexist,
		     uid_t *new, mach_msg_type_number_t nnew)
    {
      error_t urp;
      int i, j, k;
      vm_size_t offset;

      urp = __vm_allocate (mach_task_self (), (vm_address_t *) newlistp,
			   nexist + nnew * sizeof (uid_t), 1);
      if (urp)
	return urp;

      j = 0;
      for (i = 0; i < nexist; i++)
	(*newlistp)[j++] = exist[i];

      for (i = 0; i < nnew; i++)
	{
	  for (k = 0; k < nexist; k++)
	    if (exist[k] == new[i])
	      break;
	  if (k < nexist)
	    continue;

	  (*newlistp)[j++] = new[i];
	}

      offset = (round_page (nexist + nnew * sizeof (uid_t))
		- round_page (j * sizeof (uid_t)));
      if (offset)
	__vm_deallocate (mach_task_self (),
		         (vm_address_t) (*newlistp
				         + (nexist + nnew * sizeof (uid_t))),
		         offset);
      *newlistlen = j;
      return 0;
    }

  /* Find out what ids ADDAUTH refers to */

  genuids = gengids = auxuids = auxgids = 0;
  ngenuids = ngengids = nauxuids = nauxgids = 0;
  err = __auth_getids (addauth,
		       &genuids, &ngenuids,
		       &auxuids, &nauxuids,
		       &gengids, &ngengids,
		       &auxgids, &nauxgids);
  if (err)
    return err;

  /* OR in these ids to what we already have, creating a new list. */

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);
  _hurd_check_ids ();

#define MAKE(genaux,uidgid) 						    \
  make_list (&new ## genaux ## uidgid ## s, 				    \
	     &nnew ## genaux ## uidgid ## s,				    \
	     _hurd_id.genaux.uidgid ## s,				    \
	     _hurd_id.genaux.n ## uidgid ## s,				    \
	     genaux ## uidgid ## s,					    \
	     n ## genaux ## uidgid ## s)

  err = MAKE (gen, uid);
  if (!err)
    MAKE (aux, uid);
  if (!err)
    MAKE (gen, gid);
  if (!err)
    MAKE (aux, gid);
#undef MAKE

  __mutex_unlock (&_hurd_id.lock);
  HURD_CRITICAL_END;


  /* Create the new auth port */

  if (!err)
    err = __USEPORT (AUTH,
		     __auth_makeauth (port,
				      &addauth, MACH_MSG_TYPE_MOVE_SEND, 1,
				      newgenuids, nnewgenuids,
				      newauxuids, nnewauxuids,
				      newgengids, nnewgengids,
				      newauxgids, nnewauxgids,
				      &newauth));

#define freeup(array, len) \
  if (array) \
    __vm_deallocate (mach_task_self (), (vm_address_t) array, \
		     len * sizeof (uid_t));

  freeup (genuids, ngenuids);
  freeup (auxuids, nauxuids);
  freeup (gengids, ngengids);
  freeup (auxgids, nauxgids);
  freeup (newgenuids, nnewgenuids);
  freeup (newauxuids, nnewauxuids);
  freeup (newgengids, nnewgengids);
  freeup (newauxgids, nnewauxgids);
#undef freeup

  if (err)
    return err;

  /* And install it. */

  err = __setauth (newauth);
  __mach_port_deallocate (__mach_task_self (), newauth);
  if (err)
    return errno;

  return 0;
}

kern_return_t
_S_msg_del_auth (mach_port_t me,
		 task_t task,
		 intarray_t uids, mach_msg_type_number_t nuids,
		 intarray_t gids, mach_msg_type_number_t ngids)
{
  error_t err;
  auth_t newauth;

  if (!_hurd_refport_secure_p (task))
    return EPERM;

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_id.lock);
  err = _hurd_check_ids ();

  if (!err)
    {
      size_t i, j;
      size_t nu = _hurd_id.gen.nuids, ng = _hurd_id.gen.ngids;
      uid_t newu[nu];
      gid_t newg[ng];

      memcpy (newu, _hurd_id.gen.uids, nu * sizeof (uid_t));
      memcpy (newg, _hurd_id.gen.gids, ng * sizeof (gid_t));

      for (j = 0; j < nuids; ++j)
	{
	  const uid_t uid = uids[j];
	  for (i = 0; i < nu; ++i)
	    if (newu[i] == uid)
	      /* Move the last uid into this slot, and decrease the
		 number of uids so the last slot is no longer used.  */
	      newu[i] = newu[--nu];
	}
      __vm_deallocate (__mach_task_self (),
		       (vm_address_t) uids, nuids * sizeof (uid_t));

      for (j = 0; j < ngids; ++j)
	{
	  const gid_t gid = gids[j];
	  for (i = 0; i < nu; ++i)
	    if (newu[i] == gid)
	      /* Move the last gid into this slot, and decrease the
		 number of gids so the last slot is no longer used.  */
	      newu[i] = newu[--nu];
	}
      __vm_deallocate (__mach_task_self (),
		       (vm_address_t) gids, ngids * sizeof (gid_t));

      err = __USEPORT (AUTH, __auth_makeauth
		       (port,
			NULL, MACH_MSG_TYPE_COPY_SEND, 0,
			newu, nu,
			_hurd_id.aux.uids, _hurd_id.aux.nuids,
			newg, ng,
			_hurd_id.aux.uids, _hurd_id.aux.ngids,
			&newauth));
    }
  __mutex_unlock (&_hurd_id.lock);
  HURD_CRITICAL_END;

  if (err)
    return err;

  err = __setauth (newauth);
  __mach_port_deallocate (__mach_task_self (), newauth);
  if (err)
    return errno;

  return 0;
}
