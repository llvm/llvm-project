/* Detach to target process.
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

#include <stdlib.h>

#include "thread_dbP.h"


td_err_e
td_ta_delete (td_thragent_t *ta)
{
  LOG ("td_ta_delete");

  /* Safety check.  Note that the test will also fail for TA == NULL.  */
  if (!ta_ok (ta))
    return TD_BADTA;

  /* Remove the handle from the list.  */
  list_del (&ta->list);

  /* The handle was allocated in `td_ta_new'.  */
  free (ta);

  return TD_OK;
}
