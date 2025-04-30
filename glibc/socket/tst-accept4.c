/* Test the accept4 function with differing flags arguments.
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

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <support/check.h>
#include <support/xsocket.h>
#include <support/xunistd.h>
#include <sys/socket.h>

static bool
is_nonblocking (int fd)
{
  int status = fcntl (fd, F_GETFL);
  if (status < 0)
    FAIL_EXIT1 ("fcntl (F_GETFL): %m");
  return status & O_NONBLOCK;
}

static bool
is_cloexec (int fd)
{
  int status = fcntl (fd, F_GETFD);
  if (status < 0)
    FAIL_EXIT1 ("fcntl (F_GETFD): %m");
  return status & FD_CLOEXEC;
}

struct client
{
  int socket;
  struct sockaddr_in address;
};

/* Perform a non-blocking connect to *SERVER_ADDRESS.  */
static struct client
client_connect (const struct sockaddr_in *server_address)
{
  struct client result;
  result.socket = xsocket (AF_INET,
                           SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  TEST_VERIFY (is_nonblocking (result.socket));
  TEST_VERIFY (is_cloexec (result.socket));
  int ret = connect (result.socket, (const struct sockaddr *) server_address,
                     sizeof (*server_address));
  if (ret < 0 && errno != EINPROGRESS)
    FAIL_EXIT1 ("client connect: %m");
  socklen_t sa_len = sizeof (result.address);
  xgetsockname (result.socket, (struct sockaddr *) &result.address,
                &sa_len);
  TEST_VERIFY (sa_len == sizeof (result.address));
  return result;
}

static void
check_same_address (const struct sockaddr_in *left,
                    const struct sockaddr_in *right)
{
  TEST_VERIFY (left->sin_family == AF_INET);
  TEST_VERIFY (right->sin_family == AF_INET);
  TEST_VERIFY (left->sin_addr.s_addr == right->sin_addr.s_addr);
  TEST_VERIFY (left->sin_port == right->sin_port);
}

static int
do_test (void)
{
  /* Create server socket.  */
  int server_socket = xsocket (AF_INET, SOCK_STREAM, 0);
  TEST_VERIFY (!is_nonblocking (server_socket));
  TEST_VERIFY (!is_cloexec (server_socket));
  struct sockaddr_in server_address =
    {
      .sin_family = AF_INET,
      .sin_addr = {.s_addr = htonl (INADDR_LOOPBACK) },
    };
  xbind (server_socket,
         (struct sockaddr *) &server_address, sizeof (server_address));
  {
    socklen_t sa_len = sizeof (server_address);
    xgetsockname (server_socket, (struct sockaddr *) &server_address,
                  &sa_len);
    TEST_VERIFY (sa_len == sizeof (server_address));
  }
  xlisten (server_socket, 5);

  for (int do_nonblock = 0; do_nonblock < 2; ++do_nonblock)
    for (int do_cloexec = 0; do_cloexec < 2; ++do_cloexec)
      {
        int sockflags = 0;
        if (do_nonblock)
          sockflags |= SOCK_NONBLOCK;
        if (do_cloexec)
          sockflags |= SOCK_CLOEXEC;

        struct client client = client_connect (&server_address);
        struct sockaddr_in client_address;
        socklen_t sa_len = sizeof (client_address);
        int client_socket = xaccept4 (server_socket,
                                      (struct sockaddr *) &client_address,
                                      &sa_len, sockflags);
        TEST_VERIFY (sa_len == sizeof (client_address));
        TEST_VERIFY (is_nonblocking (client_socket) == do_nonblock);
        TEST_VERIFY (is_cloexec (client_socket) == do_cloexec);
        check_same_address (&client.address, &client_address);
        xclose (client_socket);
        xclose (client.socket);
      }

  xclose (server_socket);
  return 0;
}

#include <support/test-driver.c>
