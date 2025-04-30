/* Retrieve event.
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

#include <stddef.h>
#include <string.h>

#include "thread_dbP.h"


td_err_e
td_ta_event_getmsg (const td_thragent_t *ta_arg, td_event_msg_t *msg)
{
  td_thragent_t *const ta = (td_thragent_t *) ta_arg;
  td_err_e err;
  psaddr_t eventbuf, eventnum, eventdata;
  psaddr_t thp, next;
  void *copy;

  /* XXX I cannot think of another way but using a static variable.  */
  /* XXX Use at least __thread once it is possible.  */
  static td_thrhandle_t th;

  LOG ("td_thr_event_getmsg");

  /* Test whether the TA parameter is ok.  */
  if (! ta_ok (ta))
    return TD_BADTA;

  /* Get the pointer to the thread descriptor with the last event.  */
  err = DB_GET_VALUE (thp, ta, __nptl_last_event, 0);
  if (err != TD_OK)
    return err;

  if (thp == 0)
    /* Nothing waiting.  */
    return TD_NOMSG;

  /* Copy the event message buffer in from the inferior.  */
  err = DB_GET_FIELD_ADDRESS (eventbuf, ta, thp, pthread, eventbuf, 0);
  if (err == TD_OK)
    err = DB_GET_STRUCT (copy, ta, eventbuf, td_eventbuf_t);
  if (err != TD_OK)
    return err;

  /* Read the event details from the target thread.  */
  err = DB_GET_FIELD_LOCAL (eventnum, ta, copy, td_eventbuf_t, eventnum, 0);
  if (err != TD_OK)
    return err;
  /* If the structure is on the list there better be an event recorded.  */
  if ((int) (uintptr_t) eventnum == TD_EVENT_NONE)
    return TD_DBERR;

  /* Fill the user's data structure.  */
  err = DB_GET_FIELD_LOCAL (eventdata, ta, copy, td_eventbuf_t, eventdata, 0);
  if (err != TD_OK)
    return err;

  /* Generate the thread descriptor.  */
  th.th_ta_p = (td_thragent_t *) ta;
  th.th_unique = thp;

  /* Fill the user's data structure.  */
  msg->msg.data = (uintptr_t) eventdata;
  msg->event = (uintptr_t) eventnum;
  msg->th_p = &th;

  /* And clear the event message in the target.  */
  memset (copy, 0, ta->ta_sizeof_td_eventbuf_t);
  err = DB_PUT_STRUCT (ta, eventbuf, td_eventbuf_t, copy);
  if (err != TD_OK)
    return err;

  /* Get the pointer to the next descriptor with an event.  */
  err = DB_GET_FIELD (next, ta, thp, pthread, nextevent, 0);
  if (err != TD_OK)
    return err;

  /* Store the pointer in the list head variable.  */
  err = DB_PUT_VALUE (ta, __nptl_last_event, 0, next);
  if (err != TD_OK)
    return err;

  if (next != 0)
    /* Clear the next pointer in the current descriptor.  */
    err = DB_PUT_FIELD (ta, thp, pthread, nextevent, 0, 0);

  return err;
}
