/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1997.

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

#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"

void
nis_ping (const_nis_name dirname, unsigned int utime,
	  const nis_object *dirobj)
{
  nis_result *res = NULL;
  nis_object *obj;
  ping_args args;
  unsigned int i;

  if (dirname == NULL && dirobj == NULL)
    abort ();

  if (dirobj == NULL)
    {
      res = nis_lookup (dirname, MASTER_ONLY);
      if (res == NULL || NIS_RES_STATUS (res) != NIS_SUCCESS)
	{
	  nis_freeresult (res);
	  return;
	}
      obj = res->objects.objects_val;
    }
  else
    obj = (nis_object *) dirobj;

  /* Check if obj is really a diryectory object */
  if (__type_of (obj) != NIS_DIRECTORY_OBJ)
    {
      nis_freeresult (res);
      return;
    }

  if (dirname == NULL)
    args.dir = obj->DI_data.do_name;
  else
    args.dir = (char *) dirname;
  args.stamp = utime;

  /* Send the ping only to replicas */
  for (i = 1; i < obj->DI_data.do_servers.do_servers_len; ++i)
    __do_niscall2 (&obj->DI_data.do_servers.do_servers_val[i], 1,
		   NIS_PING, (xdrproc_t) _xdr_ping_args,
		   (caddr_t) &args, (xdrproc_t) xdr_void,
		   (caddr_t) NULL, 0, NULL);
  nis_freeresult (res);
}
libnsl_hidden_nolink_def (nis_ping, GLIBC_2_1)
