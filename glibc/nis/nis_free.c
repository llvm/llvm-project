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

void
__free_fdresult (fd_result *res)
{
  if (res != NULL)
    {
      xdr_free ((xdrproc_t)_xdr_fd_result, (char *)res);
      free (res);
    }
}
libnsl_hidden_nolink_def (__free_fdresult, GLIBC_2_1)

void
nis_free_request (ib_request *ibreq)
{
  if (ibreq != NULL)
    {
      xdr_free ((xdrproc_t)_xdr_ib_request, (char *)ibreq);
      free (ibreq);
    }
}
libnsl_hidden_nolink_def (nis_free_request, GLIBC_2_1)

void
nis_free_directory (directory_obj *obj)
{
  if (obj != NULL)
    {
      xdr_free ((xdrproc_t)_xdr_directory_obj, (char *)obj);
      free (obj);
    }
}
libnsl_hidden_nolink_def (nis_free_directory, GLIBC_2_1)

void
nis_free_object (nis_object *obj)
{
  if (obj != NULL)
    {
      xdr_free ((xdrproc_t)_xdr_nis_object, (char *)obj);
      free (obj);
    }
}
libnsl_hidden_nolink_def (nis_free_object, GLIBC_2_1)

void
nis_freeresult (nis_result *res)
{
  if (res != NULL)
    {
      xdr_free ((xdrproc_t)_xdr_nis_result, (char *)res);
      free (res);
    }
}
libnsl_hidden_nolink_def (nis_freeresult, GLIBC_2_1)
