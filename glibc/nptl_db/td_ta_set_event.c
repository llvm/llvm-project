/* Globally enable events.
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

#include <stdint.h>
#include "thread_dbP.h"


td_err_e
td_ta_set_event (const td_thragent_t *ta_arg, td_thr_events_t *event)
{
  td_thragent_t *const ta = (td_thragent_t *) ta_arg;
  td_err_e err;
  psaddr_t eventmask = 0;
  void *copy = NULL;

  LOG ("td_ta_set_event");

  /* Test whether the TA parameter is ok.  */
  if (! ta_ok (ta))
    return TD_BADTA;

  /* Fetch the old event mask from the inferior and modify it in place.  */
  err = DB_GET_SYMBOL (eventmask, ta, __nptl_threads_events);
  if (err == TD_OK)
    err = DB_GET_STRUCT (copy, ta, eventmask, td_thr_events_t);
  if (err == TD_OK)
    {
      uint32_t idx;
      for (idx = 0; idx < TD_EVENTSIZE; ++idx)
	{
	  psaddr_t word;
	  uint32_t mask;
	  err = DB_GET_FIELD_LOCAL (word, ta, copy,
				    td_thr_events_t, event_bits, idx);
	  if (err != TD_OK)
	    break;
	  mask = (uintptr_t) word;
	  mask |= event->event_bits[idx];
	  word = (psaddr_t) (uintptr_t) mask;
	  err = DB_PUT_FIELD_LOCAL (ta, copy,
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
	err = DB_PUT_STRUCT (ta, eventmask, td_thr_events_t, copy);
    }

  return err;
}
