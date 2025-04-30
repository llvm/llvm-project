/* Enable full ICMP errors on a socket.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
#include <netinet/in.h>
#include <sys/socket.h>

int
__res_enable_icmp (int family, int fd)
{
  int one = 1;
  switch (family)
    {
    case AF_INET:
      return __setsockopt (fd, SOL_IP, IP_RECVERR, &one, sizeof (one));
    case AF_INET6:
      return __setsockopt (fd, SOL_IPV6, IPV6_RECVERR, &one, sizeof (one));
    default:
      __set_errno (EAFNOSUPPORT);
      return -1;
    }
}
