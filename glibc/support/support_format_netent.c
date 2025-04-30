/* Convert a struct netent object to a string.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/format_nss.h>

#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/support.h>
#include <support/xmemstream.h>

char *
support_format_netent (struct netent *e)
{
  if (e == NULL)
    {
      char *value = support_format_herrno (h_errno);
      char *result = xasprintf ("error: %s\n", value);
      free (value);
      return result;
    }

  struct xmemstream mem;
  xopen_memstream (&mem);

  if (e->n_name != NULL)
    fprintf (mem.out, "name: %s\n", e->n_name);
  for (char **ap = e->n_aliases; *ap != NULL; ++ap)
    fprintf (mem.out, "alias: %s\n", *ap);
  if (e->n_addrtype != AF_INET)
    fprintf (mem.out, "addrtype: %d\n", e->n_addrtype);
  /* On alpha, e->n_net is an unsigned long.  */
  unsigned int n_net = e->n_net;
  fprintf (mem.out, "net: 0x%08x\n", n_net);

  xfclose_memstream (&mem);
  return mem.buffer;
}
