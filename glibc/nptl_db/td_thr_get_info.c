/* Get thread information.
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
#include <stdint.h>
#include "thread_dbP.h"


td_err_e
td_thr_get_info (const td_thrhandle_t *th, td_thrinfo_t *infop)
{
  td_err_e err;
  void *copy;
  psaddr_t tls, schedpolicy, schedprio, cancelhandling, tid, report_events;

  LOG ("td_thr_get_info");

  if (th->th_unique == 0)
    {
      /* Special case for the main thread before initialization.  */
      copy = NULL;
      tls = 0;
      cancelhandling = 0;
      schedpolicy = SCHED_OTHER;
      schedprio = 0;
      tid = 0;

      /* Ignore errors to obtain the __nptl_initial_report_events
	 value because GDB no longer uses the events interface, and
	 other libthread_db consumers hopefully can handle different
	 libpthread/lds.o load orders.  */
      report_events = 0;
      (void) DB_GET_VALUE (report_events, th->th_ta_p,
			   __nptl_initial_report_events, 0);
      err = TD_OK;
    }
  else
    {
      /* Copy the whole descriptor in once so we can access the several
	 fields locally.  Excess copying in one go is much better than
	 multiple ps_pdread calls.  */
      err = DB_GET_STRUCT (copy, th->th_ta_p, th->th_unique, pthread);
      if (err != TD_OK)
	return err;

      err = DB_GET_FIELD_ADDRESS (tls, th->th_ta_p, th->th_unique,
				  pthread, specific, 0);
      if (err != TD_OK)
	return err;

      err = DB_GET_FIELD_LOCAL (schedpolicy, th->th_ta_p, copy, pthread,
				schedpolicy, 0);
      if (err != TD_OK)
	return err;
      err = DB_GET_FIELD_LOCAL (schedprio, th->th_ta_p, copy, pthread,
				schedparam_sched_priority, 0);
      if (err != TD_OK)
	return err;
      err = DB_GET_FIELD_LOCAL (tid, th->th_ta_p, copy, pthread, tid, 0);
      if (err != TD_OK)
	return err;
      err = DB_GET_FIELD_LOCAL (cancelhandling, th->th_ta_p, copy, pthread,
				cancelhandling, 0);
      if (err != TD_OK)
	return err;
      err = DB_GET_FIELD_LOCAL (report_events, th->th_ta_p, copy, pthread,
				report_events, 0);
    }
  if (err != TD_OK)
    return err;

  /* Fill in information.  Clear first to provide reproducable
     results for the fields we do not fill in.  */
  memset (infop, '\0', sizeof (td_thrinfo_t));

  infop->ti_tid = (thread_t) th->th_unique;
  infop->ti_tls = (char *) tls;
  infop->ti_pri = ((uintptr_t) schedpolicy == SCHED_OTHER
		   ? 0 : (uintptr_t) schedprio);
  infop->ti_type = TD_THR_USER;

  if ((((int) (uintptr_t) cancelhandling) & EXITING_BITMASK) == 0)
    /* XXX For now there is no way to get more information.  */
    infop->ti_state = TD_THR_ACTIVE;
  else if ((((int) (uintptr_t) cancelhandling) & TERMINATED_BITMASK) == 0)
    infop->ti_state = TD_THR_ZOMBIE;
  else
    infop->ti_state = TD_THR_UNKNOWN;

  /* Initialization which are the same in both cases.  */
  infop->ti_ta_p = th->th_ta_p;
  infop->ti_lid = tid == 0 ? ps_getpid (th->th_ta_p->ph) : (uintptr_t) tid;
  infop->ti_traceme = report_events != 0;

  if (copy != NULL)
    err = DB_GET_FIELD_LOCAL (infop->ti_startfunc, th->th_ta_p, copy, pthread,
			      start_routine, 0);
  if (copy != NULL && err == TD_OK)
    {
      uint32_t idx;
      for (idx = 0; idx < TD_EVENTSIZE; ++idx)
	{
	  psaddr_t word;
	  err = DB_GET_FIELD_LOCAL (word, th->th_ta_p, copy, pthread,
				    eventbuf_eventmask_event_bits, idx);
	  if (err != TD_OK)
	    break;
	  infop->ti_events.event_bits[idx] = (uintptr_t) word;
	}
      if (err == TD_NOAPLIC)
	memset (&infop->ti_events.event_bits[idx], 0,
		(TD_EVENTSIZE - idx) * sizeof infop->ti_events.event_bits[0]);
    }

  return err;
}
