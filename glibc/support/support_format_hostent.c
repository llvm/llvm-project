/* Convert a struct hostent object to a string.
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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/support.h>
#include <support/xmemstream.h>

static int
address_length (int family)
{
  switch (family)
    {
    case AF_INET:
      return 4;
    case AF_INET6:
      return 16;
    }
  return -1;
}

char *
support_format_hostent (struct hostent *h)
{
  if (h == NULL)
    {
      if (h_errno == NETDB_INTERNAL)
        return xasprintf ("error: NETDB_INTERNAL (errno %d, %m)\n", errno);
      else
        {
          char *value = support_format_herrno (h_errno);
          char *result = xasprintf ("error: %s\n", value);
          free (value);
          return result;
        }
    }

  struct xmemstream mem;
  xopen_memstream (&mem);

  fprintf (mem.out, "name: %s\n", h->h_name);
  for (char **alias = h->h_aliases; *alias != NULL; ++alias)
    fprintf (mem.out, "alias: %s\n", *alias);
  for (unsigned i = 0; h->h_addr_list[i] != NULL; ++i)
    {
      char buf[128];
      if (inet_ntop (h->h_addrtype, h->h_addr_list[i],
                     buf, sizeof (buf)) == NULL)
        fprintf (mem.out, "error: inet_ntop failed: %m\n");
      else
        fprintf (mem.out, "address: %s\n", buf);
    }
  if (h->h_length != address_length (h->h_addrtype))
    {
      char *family = support_format_address_family (h->h_addrtype);
      fprintf (mem.out, "error: invalid address length %d for %s\n",
               h->h_length, family);
      free (family);
    }

  xfclose_memstream (&mem);
  return mem.buffer;
}
