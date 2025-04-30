/* Test listing of network interface addresses.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static int failures;

static const char *
addr_string (struct sockaddr *sa, char *buf, size_t size)
{
  if (sa == NULL)
    return "<none>";

  switch (sa->sa_family)
    {
    case AF_INET:
      return inet_ntop (AF_INET, &((struct sockaddr_in *) sa)->sin_addr,
			buf, size);
    case AF_INET6:
      return inet_ntop (AF_INET6, &((struct sockaddr_in6 *) sa)->sin6_addr,
			buf, size);
#ifdef AF_LINK
    case AF_LINK:
      return "<link>";
#endif
    case AF_UNSPEC:
      return "---";

#ifdef AF_PACKET
    case AF_PACKET:
      return "<packet>";
#endif

    default:
      ++failures;
      printf ("sa_family=%d %08x\n", sa->sa_family,
	      *(int*)&((struct sockaddr_in *) sa)->sin_addr.s_addr);
      return "<unexpected sockaddr family>";
    }
}


static int
do_test (void)
{
  struct ifaddrs *ifaces, *ifa;

  if (getifaddrs (&ifaces) < 0)
    {
      if (errno != ENOSYS)
	{
	  printf ("Couldn't get any interfaces: %s.\n", strerror (errno));
	  exit (1);
	}
      /* The function is simply not implemented.  */
      exit (0);
    }

  puts ("\
Name           Flags   Address         Netmask         Broadcast/Destination");

  for (ifa = ifaces; ifa != NULL; ifa = ifa->ifa_next)
    {
      char abuf[64], mbuf[64], dbuf[64];
      printf ("%-15s%#.4x  %-15s %-15s %-15s\n",
	      ifa->ifa_name, ifa->ifa_flags,
	      addr_string (ifa->ifa_addr, abuf, sizeof (abuf)),
	      addr_string (ifa->ifa_netmask, mbuf, sizeof (mbuf)),
	      addr_string (ifa->ifa_broadaddr, dbuf, sizeof (dbuf)));
    }

  freeifaddrs (ifaces);

  return failures ? 1 : 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
