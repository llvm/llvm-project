/* Enable event process-wide.
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
td_thr_event_enable (const td_thrhandle_t *th, int onoff)
{
  LOG ("td_thr_event_enable");

  if (th->th_unique != 0)
    {
      /* Write the new value into the thread data structure.  */
      td_err_e err = DB_PUT_FIELD (th->th_ta_p, th->th_unique, pthread,
				   report_events, 0,
				   (psaddr_t) 0 + (onoff != 0));
      if (err != TD_OK)
	return err;

      /* Just in case we are in the window between initializing __stack_user
	 and copying from __nptl_initial_report_events, we set it too.
	 It doesn't hurt to do this for non-initial threads, since it
	 won't be consulted again anyway.  It would take another fetch
	 to get the tid and determine this isn't the initial thread,
	 so just do it always.  */
    }

  /* We are faking it for the initial thread before its thread
     descriptor is set up.  */
  return DB_PUT_VALUE (th->th_ta_p, __nptl_initial_report_events, 0,
		       (psaddr_t) 0 + (onoff != 0));
}
