/* Smoke test for socket options.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <support/xsocket.h>
#include <support/xunistd.h>
#include <support/check.h>
#include <netinet/in.h>

static int
do_test (void)
{
  int fd = xsocket (AF_INET, SOCK_STREAM, IPPROTO_TCP);

  struct linger value = { -1, -1 };
  socklen_t optlen = sizeof (value);
  TEST_COMPARE (getsockopt (fd, SOL_SOCKET, SO_LINGER, &value, &optlen), 0);
  TEST_COMPARE (optlen, sizeof (value));
  TEST_COMPARE (value.l_onoff, 0);
  TEST_COMPARE (value.l_linger, 0);

  value.l_onoff = 1;
  value.l_linger = 30;
  TEST_COMPARE (setsockopt (fd, SOL_SOCKET, SO_LINGER, &value, sizeof (value)),
                0);

  value.l_onoff = -1;
  value.l_linger = -1;
  TEST_COMPARE (getsockopt (fd, SOL_SOCKET, SO_LINGER, &value, &optlen), 0);
  TEST_COMPARE (optlen, sizeof (value));
  TEST_COMPARE (value.l_onoff, 1);
  TEST_COMPARE (value.l_linger, 30);

  xclose (fd);
  return 0;
}

#include <support/test-driver.c>
