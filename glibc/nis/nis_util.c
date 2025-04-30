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

#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"

fd_result *
__nis_finddirectory (directory_obj *dir, const_nis_name name)
{
  nis_error status;
  fd_args fd_args;
  fd_result *fd_res;

  fd_args.dir_name = (char *)name;
  fd_args.requester = nis_local_host();
  fd_res = calloc (1, sizeof (fd_result));
  if (fd_res == NULL)
    return NULL;

  status = __do_niscall2 (dir->do_servers.do_servers_val,
			  dir->do_servers.do_servers_len,
			  NIS_FINDDIRECTORY, (xdrproc_t) _xdr_fd_args,
			  (caddr_t) &fd_args, (xdrproc_t) _xdr_fd_result,
			  (caddr_t) fd_res, NO_AUTHINFO|USE_DGRAM, NULL);
  if (status != NIS_SUCCESS)
    fd_res->status = status;

  return fd_res;
}
libnsl_hidden_nolink_def (__nis_finddirectory, GLIBC_2_1)

/* The hash implementation is in a separate file.  */
#include "nis_hash.c"
