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
#include <rpc/rpc.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"

nis_object *
nis_clone_object (const nis_object *src, nis_object *dest)
{
  char *addr;
  unsigned int size;
  XDR xdrs;
  nis_object *res = NULL;

  if (src == NULL)
    return (NULL);

  size = xdr_sizeof ((xdrproc_t)_xdr_nis_object, (char *) src);
  if ((addr = calloc (1, size)) == NULL)
    return NULL;

  if (dest == NULL)
    {
      if ((res = calloc (1, sizeof (nis_object))) == NULL)
	goto out;
    }
  else
    res = dest;

  xdrmem_create (&xdrs, addr, size, XDR_ENCODE);
  if (!_xdr_nis_object (&xdrs, (nis_object *) src))
    goto out2;
  xdr_destroy (&xdrs);
  xdrmem_create (&xdrs, addr, size, XDR_DECODE);
  if (!_xdr_nis_object (&xdrs, res))
    {
    out2:
      if (dest == NULL)
	free (res);
      res = NULL;
    }

  xdr_destroy (&xdrs);
 out:
  free (addr);

  return res;
}
libnsl_hidden_nolink_def (nis_clone_object, GLIBC_2_1)
