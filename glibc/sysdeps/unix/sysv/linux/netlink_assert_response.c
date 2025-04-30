/* Check recvmsg results for netlink sockets.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <fcntl.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/socket.h>

#include "netlinkaccess.h"

static int
get_address_family (int fd)
{
  struct sockaddr_storage sa;
  socklen_t sa_len = sizeof (sa);
  if (__getsockname (fd, (struct sockaddr *) &sa, &sa_len) < 0)
    return -1;
  /* Check that the socket family number is preserved despite in-band
     signaling.  */
  _Static_assert (sizeof (sa.ss_family) < sizeof (int), "address family size");
  _Static_assert (0 < (__typeof__ (sa.ss_family)) -1,
                  "address family unsigned");
  return sa.ss_family;
}

void
__netlink_assert_response (int fd, ssize_t result)
{
  if (result < 0)
    {
      /* Check if the error is unexpected.  */
      bool terminate = false;
      int error_code = errno;
      int family = get_address_family (fd);
      if (family != AF_NETLINK)
        /* If the address family does not match (or getsockname
           failed), report the original error.  */
        terminate = true;
      else if (error_code == EBADF
          || error_code == ENOTCONN
          || error_code == ENOTSOCK
          || error_code == ECONNREFUSED)
        /* These errors indicate that the descriptor is not a
           connected socket.  */
        terminate = true;
      else if (error_code == EAGAIN || error_code == EWOULDBLOCK)
        {
          /* The kernel might return EAGAIN for other reasons than a
             non-blocking socket.  But if the socket is not blocking,
             it is not ours, so report the error.  */
          int mode = __fcntl (fd, F_GETFL, 0);
          if (mode < 0 || (mode & O_NONBLOCK) != 0)
            terminate = true;
        }
      if (terminate)
        {
          char message[200];
          if (family < 0)
            __snprintf (message, sizeof (message),
                        "Unexpected error %d on netlink descriptor %d.\n",
                        error_code, fd);
          else
            __snprintf (message, sizeof (message),
                        "Unexpected error %d on netlink descriptor %d"
                        " (address family %d).\n",
                        error_code, fd, family);
          __libc_fatal (message);
        }
      else
        /* Restore orignal errno value.  */
        __set_errno (error_code);
    }
  else if (result < sizeof (struct nlmsghdr))
    {
      char message[200];
      int family = get_address_family (fd);
      if (family < 0)
          __snprintf (message, sizeof (message),
                      "Unexpected netlink response of size %zd"
                      " on descriptor %d\n",
                      result, fd);
      else
          __snprintf (message, sizeof (message),
                      "Unexpected netlink response of size %zd"
                      " on descriptor %d (address family %d)\n",
                      result, fd, family);
      __libc_fatal (message);
    }
}
libc_hidden_def (__netlink_assert_response)
