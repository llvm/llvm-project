/* Attach to target process.
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
#include <stdlib.h>
#include <string.h>
#include <version.h>

#include "thread_dbP.h"


/* Datatype for the list of known thread agents.  Normally there will
   be exactly one so we don't spend much though on making it fast.  */
LIST_HEAD (__td_agent_list);


td_err_e
td_ta_new (struct ps_prochandle *ps, td_thragent_t **ta)
{
  psaddr_t versaddr;
  char versbuf[sizeof (VERSION)];

  LOG ("td_ta_new");

  /* Check whether the versions match.  */
  if (td_lookup (ps, SYM___nptl_version, &versaddr) != PS_OK)
    return TD_NOLIBTHREAD;
  if (ps_pdread (ps, versaddr, versbuf, sizeof (versbuf)) != PS_OK)
    return TD_ERR;

  if (memcmp (versbuf, VERSION, sizeof VERSION) != 0)
    /* Not the right version.  */
    return TD_VERSION;

  /* Fill in the appropriate information.  */
  *ta = (td_thragent_t *) calloc (1, sizeof (td_thragent_t));
  if (*ta == NULL)
    return TD_MALLOC;

  /* Store the proc handle which we will pass to the callback functions
     back into the debugger.  */
  (*ta)->ph = ps;

  /* Now add the new agent descriptor to the list.  */
  list_add (&(*ta)->list, &__td_agent_list);

  return TD_OK;
}
