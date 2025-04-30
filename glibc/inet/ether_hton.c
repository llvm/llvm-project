/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <errno.h>
#include <netinet/ether.h>
#include <netinet/if_ether.h>
#include <string.h>

#include "../nss/nsswitch.h"

/* Type of the lookup function we need here.  */
typedef int (*lookup_function) (const char *, struct etherent *, char *, int,
				int *);

int
ether_hostton (const char *hostname, struct ether_addr *addr)
{
  nss_action_list nip;
  union
  {
    lookup_function f;
    void *ptr;
  } fct;
  int no_more;
  enum nss_status status = NSS_STATUS_UNAVAIL;
  struct etherent etherent;

  no_more = __nss_ethers_lookup2 (&nip, "gethostton_r", NULL, &fct.ptr);

  while (no_more == 0)
    {
      char buffer[1024];

      status = (*fct.f) (hostname, &etherent, buffer, sizeof buffer, &errno);

      no_more = __nss_next2 (&nip, "gethostton_r", NULL, &fct.ptr, status, 0);
    }

  if (status == NSS_STATUS_SUCCESS)
    memcpy (addr, etherent.e_addr.ether_addr_octet,
	    sizeof (struct ether_addr));

  return status == NSS_STATUS_SUCCESS ? 0 : -1;
}
