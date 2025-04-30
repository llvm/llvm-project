/* Helper for SA_LEN macro.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <sys/socket.h>

/* If _HAVE_SA_LEN is defined, then SA_LEN just uses sockaddr.sa_len
   and there is no need for a helper function.  */

#ifndef _HAVE_SA_LEN

/* All configurations have at least these two headers
   and their associated address families.  */

# include <netinet/in.h>
# include <sys/un.h>

/* More-specific sa_len.c files #define these various HAVE_*_H
   macros and then #include this file.  */

# ifdef HAVE_NETASH_ASH_H
#  include <netash/ash.h>
# endif
# ifdef HAVE_NETATALK_AT_H
#  include <netatalk/at.h>
# endif
# ifdef HAVE_NETAX25_AX25_H
#  include <netax25/ax25.h>
# endif
# ifdef HAVE_NETECONET_EC_H
#  include <neteconet/ec.h>
# endif
# ifdef HAVE_NETIPX_IPX_H
#  include <netipx/ipx.h>
# endif
# ifdef HAVE_NETPACKET_PACKET_H
#  include <netpacket/packet.h>
# endif
# ifdef HAVE_NETROSE_ROSE_H
#  include <netrose/rose.h>
# endif
# ifdef HAVE_NETIUCV_IUCV_H
#  include <netiucv/iucv.h>
# endif

int
__libc_sa_len (sa_family_t af)
{
  switch (af)
    {
# ifdef HAVE_NETATALK_AT_H
    case AF_APPLETALK:
      return sizeof (struct sockaddr_at);
# endif
# ifdef HAVE_NETASH_ASH_H
    case AF_ASH:
      return sizeof (struct sockaddr_ash);
# endif
# ifdef HAVE_NETAX25_AX25_H
    case AF_AX25:
      return sizeof (struct sockaddr_ax25);
# endif
# ifdef HAVE_NETECONET_EC_H
    case AF_ECONET:
      return sizeof (struct sockaddr_ec);
# endif
    case AF_INET:
      return sizeof (struct sockaddr_in);
    case AF_INET6:
      return sizeof (struct sockaddr_in6);
# ifdef HAVE_NETIPX_IPX_H
    case AF_IPX:
      return sizeof (struct sockaddr_ipx);
# endif
# ifdef HAVE_NETIUCV_IUCV_H
    case AF_IUCV:
      return sizeof (struct sockaddr_iucv);
# endif
    case AF_LOCAL:
      return sizeof (struct sockaddr_un);
# ifdef HAVE_NETPACKET_PACKET_H
    case AF_PACKET:
      return sizeof (struct sockaddr_ll);
# endif
# ifdef HAVE_NETROSE_ROSE_H
    case AF_ROSE:
      return sizeof (struct sockaddr_rose);
# endif
    }
  return 0;
}
libc_hidden_def (__libc_sa_len)

#endif  /* Not _HAVE_SA_LEN.  */
