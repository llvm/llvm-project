/* Initialization function of thread debugger support library.
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

int __td_debug;


td_err_e
td_init (void)
{
  /* XXX We have to figure out what has to be done.  */
  LOG ("td_init");
  return TD_OK;
}

bool
__td_ta_rtld_global (td_thragent_t *ta)
{
  if (ta->ta_addr__rtld_global == 0)
    {
      psaddr_t rtldglobalp;
      if (DB_GET_VALUE (rtldglobalp, ta, __nptl_rtld_global, 0) == TD_OK)
        ta->ta_addr__rtld_global = rtldglobalp;
      else
        ta->ta_addr__rtld_global = (void *) -1;
    }

  return ta->ta_addr__rtld_global != (void *)-1;
}
