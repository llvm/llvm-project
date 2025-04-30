/* Test entering namespaces.
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

#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/xsocket.h>
#include <support/xunistd.h>

/* Check that the loopback interface provides multiple addresses which
   can be used to run independent servers.  */
static void
test_localhost_bind (void)
{
  printf ("info: testing loopback interface with multiple addresses\n");

  /* Create the two server addresses.  */
  static const struct addrinfo hints =
    {
      .ai_family = AF_INET,
      .ai_socktype = SOCK_DGRAM,
      .ai_protocol = IPPROTO_UDP,
    };
  struct addrinfo *ai[3];
  TEST_VERIFY_EXIT (getaddrinfo ("127.0.0.1", "53", &hints, ai + 0) == 0);
  TEST_VERIFY_EXIT (getaddrinfo ("127.0.0.2", "53", &hints, ai + 1) == 0);
  TEST_VERIFY_EXIT (getaddrinfo ("127.0.0.3", "53", &hints, ai + 2) == 0);

  /* Create the server scokets and bind them to these addresses.  */
  int sockets[3];
  for (int i = 0; i < 3; ++i)
    {
      sockets[i] = xsocket
        (ai[i]->ai_family, ai[i]->ai_socktype, ai[i]->ai_protocol);
      xbind (sockets[i], ai[i]->ai_addr, ai[i]->ai_addrlen);
    }

  /* Send two packets to each server.  */
  int client = xsocket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  for (int i = 0; i < 3; ++i)
    {
      TEST_VERIFY (sendto (client, &i, sizeof (i), 0,
                           ai[i]->ai_addr, ai[i]->ai_addrlen) == sizeof (i));
      int j = i + 256;
      TEST_VERIFY (sendto (client, &j, sizeof (j), 0,
                           ai[i]->ai_addr, ai[i]->ai_addrlen) == sizeof (j));
    }

  /* Check that the packets can be received with the expected
     contents.  Note that the receive calls interleave differently,
     which hopefully proves that the sockets are, indeed,
     independent.  */
  for (int i = 0; i < 3; ++i)
    {
      int buf;
      TEST_VERIFY (recv (sockets[i], &buf, sizeof (buf), 0) == sizeof (buf));
      TEST_VERIFY (buf == i);
    }
  for (int i = 0; i < 3; ++i)
    {
      int buf;
      TEST_VERIFY (recv (sockets[i], &buf, sizeof (buf), 0) == sizeof (buf));
      TEST_VERIFY (buf == i + 256);
      /* Check that there is no more data to receive.  */
      TEST_VERIFY (recv (sockets[i], &buf, sizeof (buf), MSG_DONTWAIT) == -1);
      TEST_VERIFY (errno == EWOULDBLOCK || errno == EAGAIN);
    }

  /* Close all sockets and free the addresses.  */
  for (int i = 0; i < 3; ++i)
    {
      freeaddrinfo (ai[i]);
      xclose (sockets[i]);
    }
  xclose (client);
}


static int
do_test (void)
{
  bool root = support_become_root ();
  if (root)
    printf ("info: acquired root-like privileges\n");
  bool netns = support_enter_network_namespace ();
  if (netns)
    printf ("info: entered network namespace\n");
  if (support_in_uts_namespace ())
    printf ("info: also entered UTS namespace\n");

  if (root && netns)
    test_localhost_bind ();

  return 0;
}

#include <support/test-driver.c>
