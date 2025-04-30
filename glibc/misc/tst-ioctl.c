/* Smoke test for ioctl.
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

#include <support/check.h>
#include <support/xunistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

static int
do_test (void)
{
  int pair[2];
  TEST_COMPARE (socketpair (AF_UNIX, SOCK_STREAM, 0, pair), 0);
  TEST_COMPARE (write (pair[0], "buffer", sizeof ("buffer")),
                sizeof ("buffer"));
  int value;
  TEST_COMPARE (ioctl (pair[1], FIONREAD, &value), 0);
  TEST_COMPARE (value, sizeof ("buffer"));
  TEST_COMPARE (ioctl (pair[0], FIONREAD, &value), 0);
  TEST_COMPARE (value, 0);
  xclose (pair[0]);
  xclose (pair[1]);
  return 0;
}

#include <support/test-driver.c>
