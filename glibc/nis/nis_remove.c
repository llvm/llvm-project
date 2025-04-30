/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@uni-paderborn.de>, 1997.

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

nis_result *
nis_remove (const_nis_name name, const nis_object *obj)
{
  nis_result *res;
  nis_error status;
  struct ns_request req;

  res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  req.ns_name = (char *)name;

  if (obj != NULL)
    {
      req.ns_object.ns_object_len = 1;
      req.ns_object.ns_object_val = nis_clone_object (obj, NULL);
    }
  else
    {
      req.ns_object.ns_object_len = 0;
      req.ns_object.ns_object_val = NULL;
    }

  if ((status = __do_niscall (name, NIS_REMOVE, (xdrproc_t) _xdr_ns_request,
			      (caddr_t) &req, (xdrproc_t) _xdr_nis_result,
			      (caddr_t) res, MASTER_ONLY,
			      NULL)) != NIS_SUCCESS)
    NIS_RES_STATUS (res) = status;

  nis_destroy_object (req.ns_object.ns_object_val);

  return res;
}
libnsl_hidden_nolink_def (nis_remove, GLIBC_2_1)
