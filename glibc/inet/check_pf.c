/* Determine protocol families for which interfaces exist.  Generic version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <ifaddrs.h>
#include <netdb.h>
#include <stdint.h>

void
attribute_hidden
__check_pf (bool *seen_ipv4, bool *seen_ipv6,
	    struct in6addrinfo **in6ai, size_t *in6ailen)
{
  /* By default we have no way to determine information about
     deprecated and temporary addresses.  */
  *in6ai = NULL;
  *in6ailen = 0;

  /* Get the interface list via getifaddrs.  */
  struct ifaddrs *ifa = NULL;
  if (__getifaddrs (&ifa) != 0)
    {
      /* We cannot determine what interfaces are available.  Be
	 pessimistic.  */
      *seen_ipv4 = true;
      *seen_ipv6 = true;
      return;
    }

  *seen_ipv4 = false;
  *seen_ipv6 = false;

  struct ifaddrs *runp;
  for (runp = ifa; runp != NULL; runp = runp->ifa_next)
    if (runp->ifa_addr->sa_family == PF_INET)
      *seen_ipv4 = true;
    else if (runp->ifa_addr->sa_family == PF_INET6)
      *seen_ipv6 = true;

  (void) __freeifaddrs (ifa);
}


void
__free_in6ai (struct in6addrinfo *in6ai)
{
  /* Nothing to do.  */
}


#if IS_IN (nscd)
uint32_t
__bump_nl_timestamp (void)
{
  return 0;
}
#endif
