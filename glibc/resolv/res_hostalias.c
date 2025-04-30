/* HOSTALIASES-based name resolution.  Public legacy functions.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <netdb.h>
#include <resolv-internal.h>
#include <resolv_context.h>

/* Common part of res_hostalias and hostalias.  */
static const char *
context_hostalias_common (struct resolv_context *ctx,
                          const char *name, char *dst, size_t siz)
{
  if (ctx == NULL)
    {
      RES_SET_H_ERRNO (&_res, NETDB_INTERNAL);
      return NULL;
    }
  const char *result = __res_context_hostalias (ctx, name, dst, siz);
  __resolv_context_put (ctx);
  return result;
}

const char *
res_hostalias (res_state statp, const char *name, char *dst, size_t siz)
{
  return context_hostalias_common
    (__resolv_context_get_override (statp), name, dst, siz);
}

const char *
hostalias (const char *name)
{
  static char abuf[MAXDNAME];
  return context_hostalias_common
    (__resolv_context_get (), name, abuf, sizeof (abuf));
}
