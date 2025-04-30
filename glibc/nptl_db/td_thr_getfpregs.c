/* Get a thread's floating-point register set.
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
td_thr_getfpregs (const td_thrhandle_t *th, prfpregset_t *regset)
{
  psaddr_t cancelhandling, tid;
  td_err_e err;

  LOG ("td_thr_getfpregs");

  if (th->th_unique == 0)
    /* Special case for the main thread before initialization.  */
    return ps_lgetfpregs (th->th_ta_p->ph, ps_getpid (th->th_ta_p->ph),
			  regset) != PS_OK ? TD_ERR : TD_OK;

  /* We have to get the state and the PID for this thread.  */
  err = DB_GET_FIELD (cancelhandling, th->th_ta_p, th->th_unique, pthread,
		      cancelhandling, 0);
  if (err != TD_OK)
    return err;

  /* If the thread already terminated we return all zeroes.  */
  if (((int) (uintptr_t) cancelhandling) & TERMINATED_BITMASK)
    memset (regset, '\0', sizeof (*regset));
  /* Otherwise get the register content through the callback.  */
  else
    {
      err = DB_GET_FIELD (tid, th->th_ta_p, th->th_unique, pthread, tid, 0);
      if (err != TD_OK)
	return err;

      if (ps_lgetfpregs (th->th_ta_p->ph, (uintptr_t) tid, regset) != PS_OK)
	return TD_ERR;
    }

  return TD_OK;
}
