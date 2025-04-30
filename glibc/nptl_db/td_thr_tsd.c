/* Get a thread-specific data pointer for a thread.
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
td_thr_tsd (const td_thrhandle_t *th, const thread_key_t tk, void **data)
{
  td_err_e err;
  psaddr_t tk_seq, level1, level2, seq, value;
  void *copy;
  uint32_t pthread_key_2ndlevel_size, idx1st, idx2nd;

  LOG ("td_thr_tsd");

  /* Get the key entry.  */
  err = DB_GET_VALUE (tk_seq, th->th_ta_p, __pthread_keys, tk);
  if (err == TD_NOAPLIC)
    return TD_BADKEY;
  if (err != TD_OK)
    return err;

  /* Fail if this key is not at all used.  */
  if (((uintptr_t) tk_seq & 1) == 0)
    return TD_BADKEY;

  /* This makes sure we have the size information on hand.  */
  err = DB_GET_FIELD_ADDRESS (level2, th->th_ta_p, 0, pthread_key_data_level2,
			      data, 1);
  if (err != TD_OK)
    return err;

  /* Compute the indices.  */
  pthread_key_2ndlevel_size
    = DB_DESC_NELEM (th->th_ta_p->ta_field_pthread_key_data_level2_data);
  idx1st = tk / pthread_key_2ndlevel_size;
  idx2nd = tk % pthread_key_2ndlevel_size;

  /* Now fetch the first level pointer.  */
  err = DB_GET_FIELD (level1, th->th_ta_p, th->th_unique, pthread,
		      specific, idx1st);
  if (err == TD_NOAPLIC)
    return TD_DBERR;
  if (err != TD_OK)
    return err;

  /* Check the pointer to the second level array.  */
  if (level1 == 0)
    return TD_NOTSD;

  /* Locate the element within the second level array.  */
  err = DB_GET_FIELD_ADDRESS (level2, th->th_ta_p,
			      level1, pthread_key_data_level2, data, idx2nd);
  if (err == TD_NOAPLIC)
    return TD_DBERR;
  if (err != TD_OK)
    return err;

  /* Now copy in that whole structure.  */
  err = DB_GET_STRUCT (copy, th->th_ta_p, level2, pthread_key_data);
  if (err != TD_OK)
    return err;

  /* Check whether the data is valid.  */
  err = DB_GET_FIELD_LOCAL (seq, th->th_ta_p, copy, pthread_key_data, seq, 0);
  if (err != TD_OK)
    return err;
  if (seq != tk_seq)
    return TD_NOTSD;

  /* Finally, fetch the value.  */
  err = DB_GET_FIELD_LOCAL (value, th->th_ta_p, copy, pthread_key_data,
			    data, 0);
  if (err == TD_OK)
    *data = value;

  return err;
}
