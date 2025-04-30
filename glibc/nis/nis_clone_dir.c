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

directory_obj *
nis_clone_directory (const directory_obj *src, directory_obj *dest)
{
  char *addr;
  unsigned int size;
  XDR xdrs;

  if (src == NULL)
    return NULL;

  size = xdr_sizeof ((xdrproc_t)_xdr_directory_obj, (char *)src);
  if ((addr = calloc(1, size)) == NULL)
    return NULL;

  xdrmem_create(&xdrs, addr, size, XDR_ENCODE);
  if (!_xdr_directory_obj (&xdrs, (directory_obj *)src))
    {
      xdr_destroy (&xdrs);
      free (addr);
      return NULL;
    }
  xdr_destroy (&xdrs);

  directory_obj *res;
  if (dest == NULL)
    {
      if ((res = calloc (1, sizeof (directory_obj))) == NULL)
	{
	  free (addr);
	  return NULL;
	}
    }
  else
    res = dest;

  xdrmem_create (&xdrs, addr, size, XDR_DECODE);
  if (!_xdr_directory_obj (&xdrs, res))
    {
      xdr_destroy (&xdrs);
      if (res != dest)
	free (res);
      free (addr);
      return NULL;
    }
  xdr_destroy (&xdrs);
  free (addr);

  return res;
}
libnsl_hidden_nolink_def(nis_clone_directory, GLIBC_2_1)
