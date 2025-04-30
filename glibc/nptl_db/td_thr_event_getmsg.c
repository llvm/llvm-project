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

#include "thread_dbP.h"
#include <assert.h>


td_err_e
td_thr_event_getmsg (const td_thrhandle_t *th, td_event_msg_t *msg)
{
  td_err_e err;
  psaddr_t eventbuf, eventnum, eventdata;
  psaddr_t thp, prevp;
  void *copy;

  LOG ("td_thr_event_getmsg");

  /* Copy the event message buffer in from the inferior.  */
  err = DB_GET_FIELD_ADDRESS (eventbuf, th->th_ta_p, th->th_unique, pthread,
			      eventbuf, 0);
  if (err == TD_OK)
    err = DB_GET_STRUCT (copy, th->th_ta_p, eventbuf, td_eventbuf_t);
  if (err != TD_OK)
    return err;

  /* Check whether an event occurred.  */
  err = DB_GET_FIELD_LOCAL (eventnum, th->th_ta_p, copy,
			    td_eventbuf_t, eventnum, 0);
  if (err != TD_OK)
    return err;
  if ((int) (uintptr_t) eventnum == TD_EVENT_NONE)
    /* Nothing.  */
    return TD_NOMSG;

  /* Fill the user's data structure.  */
  err = DB_GET_FIELD_LOCAL (eventdata, th->th_ta_p, copy,
			    td_eventbuf_t, eventdata, 0);
  if (err != TD_OK)
    return err;

  msg->msg.data = (uintptr_t) eventdata;
  msg->event = (uintptr_t) eventnum;
  msg->th_p = th;

  /* And clear the event message in the target.  */
  memset (copy, 0, th->th_ta_p->ta_sizeof_td_eventbuf_t);
  err = DB_PUT_STRUCT (th->th_ta_p, eventbuf, td_eventbuf_t, copy);
  if (err != TD_OK)
    return err;

  /* Get the pointer to the thread descriptor with the last event.
     If it doesn't match TH, then walk down the list until we find it.
     We must splice it out of the list so that there is no dangling
     pointer to it later when it dies.  */
  err = DB_GET_SYMBOL (prevp, th->th_ta_p, __nptl_last_event);
  if (err != TD_OK)
    return err;
  err = DB_GET_VALUE (thp, th->th_ta_p, __nptl_last_event, 0);
  if (err != TD_OK)
    return err;

  while (thp != 0)
    {
      psaddr_t next;
      err = DB_GET_FIELD (next, th->th_ta_p, th->th_unique, pthread,
			  nextevent, 0);
      if (err != TD_OK)
	return err;

      if (next == thp)
	return TD_DBERR;

      if (thp == th->th_unique)
	{
	  /* PREVP points at this thread, splice it out.  */
	  psaddr_t next_nextp;
	  err = DB_GET_FIELD_ADDRESS (next_nextp, th->th_ta_p, next, pthread,
				      nextevent, 0);
	  assert (err == TD_OK); /* We used this field before.  */
	  if (prevp == next_nextp)
	    return TD_DBERR;

	  err = _td_store_value (th->th_ta_p,
				 th->th_ta_p->ta_var___nptl_last_event, -1,
				 0, prevp, next);
	  if (err != TD_OK)
	    return err;

	  /* Now clear this thread's own next pointer so it's not dangling
	     when the thread resumes and then chains on for its next event.  */
	  return DB_PUT_FIELD (th->th_ta_p, thp, pthread, nextevent, 0, 0);
	}

      err = DB_GET_FIELD_ADDRESS (prevp, th->th_ta_p, thp, pthread,
				  nextevent, 0);
      assert (err == TD_OK); /* We used this field before.  */
      thp = next;
    }

  /* Ack!  This should not happen.  */
  return TD_DBERR;
}
