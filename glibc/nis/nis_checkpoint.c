/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

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

#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"

nis_result *
nis_checkpoint (const_nis_name dirname)
{
  nis_result *res;

  res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  if (dirname != NULL)
    {
      nis_result *res2;
      u_int i;

      res2 = nis_lookup (dirname, EXPAND_NAME);
      if (NIS_RES_STATUS (res2) != NIS_SUCCESS)
	{
	  free (res);
	  return res2;
	}

      /* Check if obj is really a diryectory object */
      if (__type_of (NIS_RES_OBJECT (res2)) != NIS_DIRECTORY_OBJ)
	{
	  nis_freeresult (res2);
	  NIS_RES_STATUS (res) = NIS_INVALIDOBJ;
	  return res;
	}

      for (i = 0;
	   i < NIS_RES_OBJECT (res2)->DI_data.do_servers.do_servers_len; ++i)
	{
	  cp_result cpres;

	  memset (&cpres, '\0', sizeof (cp_result));
	  if (__do_niscall2 (&NIS_RES_OBJECT(res2)->DI_data.do_servers.do_servers_val[i],
			     1, NIS_CHECKPOINT, (xdrproc_t) _xdr_nis_name,
			     (caddr_t) &dirname, (xdrproc_t) _xdr_cp_result,
			     (caddr_t) &cpres, 0, NULL) != NIS_SUCCESS)
	    NIS_RES_STATUS (res) = NIS_RPCERROR;
	  else
	    {
	      NIS_RES_STATUS (res) = cpres.cp_status;
	      res->zticks += cpres.cp_zticks;
	      res->dticks += cpres.cp_dticks;
	    }
	}
      nis_freeresult (res2);
    }
  else
    NIS_RES_STATUS (res) = NIS_NOSUCHNAME;

  return res;
}
libnsl_hidden_nolink_def (nis_checkpoint, GLIBC_2_1)
