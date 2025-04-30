/* Get address of thread local variable.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2002.

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

#include <link.h>
#include "thread_dbP.h"

td_err_e
td_thr_tls_get_addr (const td_thrhandle_t *th,
		     psaddr_t map_address, size_t offset, psaddr_t *address)
{
  td_err_e err;
  psaddr_t modid;

  /* Get the TLS module ID from the `struct link_map' in the inferior.  */
  err = DB_GET_FIELD (modid, th->th_ta_p, map_address, link_map,
		      l_tls_modid, 0);
  if (err == TD_NOCAPAB)
    return TD_NOAPLIC;
  if (err == TD_OK)
    {
      err = td_thr_tlsbase (th, (uintptr_t) modid, address);
      if (err == TD_OK)
	*address += offset;
    }
  return err;
}
