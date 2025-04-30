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

#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"

nis_error
nis_mkdir (const_nis_name dir, const nis_server *server)
{
  nis_error res, res2;

  if (server == NULL)
    res2 = __do_niscall (dir, NIS_MKDIR, (xdrproc_t) _xdr_nis_name,
			 (caddr_t) &dir, (xdrproc_t) _xdr_nis_error,
			 (caddr_t) &res, 0, NULL);
  else
    res2 = __do_niscall2 (server, 1, NIS_MKDIR,
			  (xdrproc_t) _xdr_nis_name,
			  (caddr_t) &dir, (xdrproc_t) _xdr_nis_error,
			  (caddr_t) &res, 0, NULL);
  if (res2 != NIS_SUCCESS)
    return res2;

  return res;
}
libnsl_hidden_nolink_def (nis_mkdir, GLIBC_2_1)
