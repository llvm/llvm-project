/* Test that garbage packets do not affect timeout handling.
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

#include <netinet/in.h>
#include <rpc/clnt.h>
#include <rpc/svc.h>
#include <stdbool.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/xsocket.h>
#include <support/xthread.h>
#include <sys/socket.h>
#include <unistd.h>

/* Descriptor for the server UDP socket.  */
static int server_fd;

static void *
garbage_sender_thread (void *unused)
{
  while (true)
    {
      struct sockaddr_storage sa;
      socklen_t salen = sizeof (sa);
      char buf[1];
      if (recvfrom (server_fd, buf, sizeof (buf), 0,
                    (struct sockaddr *) &sa, &salen) < 0)
        FAIL_EXIT1 ("recvfrom: %m");

      /* Send garbage packets indefinitely.  */
      buf[0] = 0;
      while (true)
        {
          /* sendto can fail if the client closed the socket.  */
          if (sendto (server_fd, buf, sizeof (buf), 0,
                      (struct sockaddr *) &sa, salen) < 0)
            break;

          /* Wait a bit, to avoid burning too many CPU cycles in a
             tight loop.  The wait period must be much shorter than
             the client timeouts configured below.  */
          usleep (50 * 1000);
        }
    }
}

static int
do_test (void)
{
  support_become_root ();
  support_enter_network_namespace ();

  server_fd = xsocket (AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);
  struct sockaddr_in server_address =
    {
      .sin_family = AF_INET,
      .sin_addr.s_addr = htonl (INADDR_LOOPBACK),
    };
  xbind (server_fd,
         (struct sockaddr *) &server_address, sizeof (server_address));
  {
    socklen_t sinlen = sizeof (server_address);
    xgetsockname (server_fd, (struct sockaddr *) &server_address, &sinlen);
    TEST_VERIFY (sizeof (server_address) == sinlen);
  }

  /* Garbage packet source.  */
  xpthread_detach (xpthread_create (NULL, garbage_sender_thread, NULL));

  /* Test client.  Use an arbitrary timeout of one second, which is
     much longer than the garbage packet interval, but still
     reasonably short, so that the test completes quickly.  */
  int client_fd = RPC_ANYSOCK;
  CLIENT *clnt = clntudp_create (&server_address,
                                 1, 2, /* Arbitrary RPC endpoint numbers.  */
                                 (struct timeval) { 1, 0 },
                                 &client_fd);
  if (clnt == NULL)
    FAIL_EXIT1 ("clntudp_create: %m");

  TEST_VERIFY (clnt_call (clnt, 3, /* Arbitrary RPC procedure number.  */
                          (xdrproc_t) xdr_void, NULL,
                          (xdrproc_t) xdr_void, NULL,
                          ((struct timeval) { 1, 0 })));

  return 0;
}

#include <support/test-driver.c>
