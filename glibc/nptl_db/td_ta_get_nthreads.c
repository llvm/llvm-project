/* Get the number of threads in the process.
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
td_ta_get_nthreads (const td_thragent_t *ta_arg, int *np)
{
  td_thragent_t *const ta = (td_thragent_t *) ta_arg;
  td_err_e err;
  psaddr_t n;

  LOG ("td_ta_get_nthreads");

  /* Test whether the TA parameter is ok.  */
  if (! ta_ok (ta))
    return TD_BADTA;

  /* Access the variable in the inferior that tells us.  */
  err = DB_GET_VALUE (n, ta, __nptl_nthreads, 0);
  if (err == TD_OK)
    *np = (uintptr_t) n;

  return err;
}
