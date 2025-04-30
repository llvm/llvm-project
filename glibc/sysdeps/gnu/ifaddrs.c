/* getifaddrs -- get names and addresses of all network interfaces
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

#include <ifaddrs.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>

#include "ifreq.h"

/* Create a linked list of `struct ifaddrs' structures, one for each
   network interface on the host machine.  If successful, store the
   list in *IFAP and return 0.  On errors, return -1 and set `errno'.  */
int
__getifaddrs (struct ifaddrs **ifap)
{
  /* This implementation handles only IPv4 interfaces.
     The various ioctls below will only work on an AF_INET socket.
     Some different mechanism entirely must be used for IPv6.  */
  int fd = __socket (AF_INET, SOCK_DGRAM, 0);
  struct ifreq *ifreqs;
  int nifs;

  if (fd < 0)
    return -1;

  __ifreq (&ifreqs, &nifs, fd);
  if (ifreqs == NULL)		/* XXX doesn't distinguish error vs none */
    {
      __close (fd);
      return -1;
    }

  /* Now we have the list of interfaces and each one's address.
     Put it into the expected format and fill in the remaining details.  */
  if (nifs == 0)
    *ifap = NULL;
  else
    {
      struct
      {
	struct ifaddrs ia;
	struct sockaddr addr, netmask, broadaddr;
	char name[IF_NAMESIZE];
      } *storage;
      struct ifreq *ifr;
      int i;

      storage = malloc (nifs * sizeof storage[0]);
      if (storage == NULL)
	{
	  __close (fd);
	  __if_freereq (ifreqs, nifs);
	  return -1;
	}

      i = 0;
      ifr = ifreqs;
      do
	{
	  /* Fill in pointers to the storage we've already allocated.  */
	  storage[i].ia.ifa_next = &storage[i + 1].ia;
	  storage[i].ia.ifa_addr = &storage[i].addr;

	  /* Now copy the information we already have from SIOCGIFCONF.  */
	  storage[i].ia.ifa_name = strncpy (storage[i].name, ifr->ifr_name,
					    sizeof storage[i].name);
	  storage[i].addr = ifr->ifr_addr;

	  /* The SIOCGIFCONF call filled in only the name and address.
	     Now we must also ask for the other information we need.  */

	  if (__ioctl (fd, SIOCGIFFLAGS, ifr) < 0)
	    break;
	  storage[i].ia.ifa_flags = ifr->ifr_flags;

	  ifr->ifr_addr = storage[i].addr;

	  if (__ioctl (fd, SIOCGIFNETMASK, ifr) < 0)
	    storage[i].ia.ifa_netmask = NULL;
	  else
	    {
	      storage[i].ia.ifa_netmask = &storage[i].netmask;
	      storage[i].netmask = ifr->ifr_netmask;
	    }

	  if (ifr->ifr_flags & IFF_BROADCAST)
	    {
	      ifr->ifr_addr = storage[i].addr;
	      if (__ioctl (fd, SIOCGIFBRDADDR, ifr) < 0)
		storage[i].ia.ifa_broadaddr = NULL;
	      {
		storage[i].ia.ifa_broadaddr = &storage[i].broadaddr;
		storage[i].broadaddr = ifr->ifr_broadaddr;
	      }
	    }
	  else if (ifr->ifr_flags & IFF_POINTOPOINT)
	    {
	      ifr->ifr_addr = storage[i].addr;
	      if (__ioctl (fd, SIOCGIFDSTADDR, ifr) < 0)
		storage[i].ia.ifa_broadaddr = NULL;
	      else
		{
		  storage[i].ia.ifa_broadaddr = &storage[i].broadaddr;
		  storage[i].broadaddr = ifr->ifr_dstaddr;
		}
	    }
	  else
	    storage[i].ia.ifa_broadaddr = NULL;

	  storage[i].ia.ifa_data = NULL; /* Nothing here for now.  */

	  ifr = __if_nextreq (ifr);
	} while (++i < nifs);
      if (i < nifs)		/* Broke out early on error.  */
	{
	  __close (fd);
	  free (storage);
	  __if_freereq (ifreqs, nifs);
	  return -1;
	}

      storage[i - 1].ia.ifa_next = NULL;

      *ifap = &storage[0].ia;

      __close (fd);
      __if_freereq (ifreqs, nifs);
    }

  return 0;
}
weak_alias (__getifaddrs, getifaddrs)
libc_hidden_def (__getifaddrs)
#ifndef getifaddrs
libc_hidden_weak (getifaddrs)
#endif

void
__freeifaddrs (struct ifaddrs *ifa)
{
  free (ifa);
}
weak_alias (__freeifaddrs, freeifaddrs)
libc_hidden_def (__freeifaddrs)
libc_hidden_weak (freeifaddrs)
