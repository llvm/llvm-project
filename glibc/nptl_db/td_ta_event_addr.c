/* Get event address.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 1999.

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

#include "thread_dbP.h"


td_err_e
td_ta_event_addr (const td_thragent_t *ta_arg,
		  td_event_e event, td_notify_t *addr)
{
  td_thragent_t *const ta = (td_thragent_t *) ta_arg;
  td_err_e err;
  psaddr_t taddr;

  LOG ("td_ta_event_addr");

  /* Test whether the TA parameter is ok.  */
  if (! ta_ok (ta))
    return TD_BADTA;

  switch (event)
    {
    case TD_CREATE:
      err = DB_GET_SYMBOL (taddr, ta, __nptl_create_event);
      break;

    case TD_DEATH:
      err = DB_GET_SYMBOL (taddr, ta, __nptl_death_event);
      break;

    default:
      /* Event cannot be handled.  */
      return TD_NOEVENT;
    }

  if (err == TD_OK)
    {
      /* Success, we got the address.  */
      addr->type = NOTIFY_BPT;
      addr->u.bptaddr = taddr;
    }

  return err;
}
