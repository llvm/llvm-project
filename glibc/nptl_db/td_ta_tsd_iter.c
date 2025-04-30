/* Iterate over a process's thread-specific data.
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
#include <alloca.h>

td_err_e
td_ta_tsd_iter (const td_thragent_t *ta_arg, td_key_iter_f *callback,
		void *cbdata_p)
{
  td_thragent_t *const ta = (td_thragent_t *) ta_arg;
  td_err_e err;
  void *keys;
  size_t keys_nb, keys_elemsize;
  psaddr_t addr;
  uint32_t idx;

  LOG ("td_ta_tsd_iter");

  /* Test whether the TA parameter is ok.  */
  if (! ta_ok (ta))
    return TD_BADTA;

  /* This makes sure we have the size information on hand.  */
  addr = 0;
  err = _td_locate_field (ta,
			  ta->ta_var___pthread_keys, SYM_DESC___pthread_keys,
			  (psaddr_t) 0 + 1, &addr);
  if (err != TD_OK)
    return err;

  /* Now copy in the entire array of key descriptors.  */
  keys_elemsize = (addr - (psaddr_t) 0) / 8;
  keys_nb = keys_elemsize * DB_DESC_NELEM (ta->ta_var___pthread_keys);
  keys = __alloca (keys_nb);
  err = DB_GET_SYMBOL (addr, ta, __pthread_keys);
  if (err != TD_OK)
    return err;
  if (ps_pdread (ta->ph, addr, keys, keys_nb) != PS_OK)
    return TD_ERR;

  /* Now get all descriptors, one after the other.  */
  for (idx = 0; idx < DB_DESC_NELEM (ta->ta_var___pthread_keys); ++idx)
    {
      psaddr_t seq, destr;
      err = DB_GET_FIELD_LOCAL (seq, ta, keys, pthread_key_struct, seq, 0);
      if (err != TD_OK)
	return err;
      if (((uintptr_t) seq) & 1)
	{
	  err = DB_GET_FIELD_LOCAL (destr, ta, keys, pthread_key_struct,
				    destr, 0);
	  if (err != TD_OK)
	    return err;
	  /* Return with an error if the callback returns a nonzero value.  */
	  if (callback ((thread_key_t) idx, destr, cbdata_p) != 0)
	    return TD_DBERR;
	}
      /* Advance to the next element in the copied array.  */
      keys += keys_elemsize;
    }

  return TD_OK;
}
