/* Validate a thread handle.
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
#include <stdbool.h>

td_err_e
__td_ta_stack_user (td_thragent_t *ta, psaddr_t *plist)
{
  if (__td_ta_rtld_global (ta))
    return DB_GET_FIELD_ADDRESS (*plist, ta, ta->ta_addr__rtld_global,
				 rtld_global, _dl_stack_user, 0);
  else
    {
      if (ta->ta_addr__dl_stack_user == 0
	  && td_mod_lookup (ta->ph, NULL, SYM__dl_stack_user,
			    &ta->ta_addr__dl_stack_user) != PS_OK)
	return TD_ERR;
      *plist = ta->ta_addr__dl_stack_user;
      return TD_OK;
    }
}

td_err_e
__td_ta_stack_used (td_thragent_t *ta, psaddr_t *plist)
{

  if (__td_ta_rtld_global (ta))
    return DB_GET_FIELD_ADDRESS (*plist, ta, ta->ta_addr__rtld_global,
				 rtld_global, _dl_stack_used, 0);
  else
    {
      if (ta->ta_addr__dl_stack_used == 0
	  && td_mod_lookup (ta->ph, NULL, SYM__dl_stack_used,
			    &ta->ta_addr__dl_stack_used) != PS_OK)
	return TD_ERR;
      *plist = ta->ta_addr__dl_stack_used;
      return TD_OK;
    }
}

static td_err_e
check_thread_list (const td_thrhandle_t *th, psaddr_t head, bool *uninit)
{
  td_err_e err;
  psaddr_t next, ofs;

  err = DB_GET_FIELD (next, th->th_ta_p, head, list_t, next, 0);
  if (err == TD_OK)
    {
      if (next == 0)
	{
	  *uninit = true;
	  return TD_NOTHR;
	}
      err = DB_GET_FIELD_ADDRESS (ofs, th->th_ta_p, 0, pthread, list, 0);
    }

  while (err == TD_OK)
    {
      if (next == head)
	return TD_NOTHR;

      if (next - (ofs - (psaddr_t) 0) == th->th_unique)
	return TD_OK;

      err = DB_GET_FIELD (next, th->th_ta_p, next, list_t, next, 0);
    }

  return err;
}


td_err_e
td_thr_validate (const td_thrhandle_t *th)
{
  td_err_e err;
  psaddr_t list;

  LOG ("td_thr_validate");

  /* First check the list with threads using user allocated stacks.  */
  bool uninit = false;
  err = __td_ta_stack_user (th->th_ta_p, &list);
  if (err == TD_OK)
    err = check_thread_list (th, list, &uninit);

  /* If our thread is not on this list search the list with stack
     using implementation allocated stacks.  */
  if (err == TD_NOTHR)
    {
      err = __td_ta_stack_used (th->th_ta_p, &list);
      if (err == TD_OK)
	err = check_thread_list (th, list, &uninit);

      if (err == TD_NOTHR && uninit && th->th_unique == 0)
	/* __pthread_initialize_minimal has not run yet.
	   There is only the special case thread handle.  */
	err = TD_OK;
    }

  return err;
}
