/* Disable specific event for thread.
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
#include <stdint.h>

#include "thread_dbP.h"


td_err_e
td_thr_clear_event (const td_thrhandle_t *th, td_thr_events_t *event)
{
  td_err_e err;
  psaddr_t eventmask;
  void *copy;

  LOG ("td_thr_clear_event");

  /* Fetch the old event mask from the inferior and modify it in place.  */
  err = DB_GET_FIELD_ADDRESS (eventmask, th->th_ta_p,
			      th->th_unique, pthread, eventbuf_eventmask, 0);
  if (err == TD_OK)
    err = DB_GET_STRUCT (copy, th->th_ta_p, eventmask, td_thr_events_t);
  if (err == TD_OK)
    {
      uint32_t idx;
      for (idx = 0; idx < TD_EVENTSIZE; ++idx)
	{
	  psaddr_t word;
	  uint32_t mask;
	  err = DB_GET_FIELD_LOCAL (word, th->th_ta_p, copy,
				    td_thr_events_t, event_bits, idx);
	  if (err != TD_OK)
	    break;
	  mask = (uintptr_t) word;
	  mask &= ~event->event_bits[idx];
	  word = (psaddr_t) (uintptr_t) mask;
	  err = DB_PUT_FIELD_LOCAL (th->th_ta_p, copy,
				    td_thr_events_t, event_bits, idx, word);
	  if (err != TD_OK)
	    break;
	}
      if (err == TD_NOAPLIC)
	{
	  err = TD_OK;
	  while (idx < TD_EVENTSIZE)
	    if (event->event_bits[idx++] != 0)
	      {
		err = TD_NOEVENT;
		break;
	      }
	}
      if (err == TD_OK)
	/* Now write it back to the inferior.  */
	err = DB_PUT_STRUCT (th->th_ta_p, eventmask, td_thr_events_t, copy);
    }

  return err;
}
