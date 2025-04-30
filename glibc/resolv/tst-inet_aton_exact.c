/* Test internal legacy IPv4 text-to-address function __inet_aton_exact.
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

#include <arpa/inet.h>
#include <support/check.h>

static int
do_test (void)
{
  struct in_addr addr = { };

  TEST_COMPARE (__inet_aton_exact ("192.0.2.1", &addr), 1);
  TEST_COMPARE (ntohl (addr.s_addr), 0xC0000201);

  TEST_COMPARE (__inet_aton_exact ("192.000.002.010", &addr), 1);
  TEST_COMPARE (ntohl (addr.s_addr), 0xC0000208);
  TEST_COMPARE (__inet_aton_exact ("0xC0000234", &addr), 1);
  TEST_COMPARE (ntohl (addr.s_addr), 0xC0000234);

  /* Trailing content is not accepted.  */
  TEST_COMPARE (__inet_aton_exact ("192.0.2.2X", &addr), 0);
  TEST_COMPARE (__inet_aton_exact ("192.0.2.3 Y", &addr), 0);
  TEST_COMPARE (__inet_aton_exact ("192.0.2.4\nZ", &addr), 0);
  TEST_COMPARE (__inet_aton_exact ("192.0.2.5\tT", &addr), 0);
  TEST_COMPARE (__inet_aton_exact ("192.0.2.6 Y", &addr), 0);
  TEST_COMPARE (__inet_aton_exact ("192.0.2.7\n", &addr), 0);
  TEST_COMPARE (__inet_aton_exact ("192.0.2.8\t", &addr), 0);

  return 0;
}

#include <support/test-driver.c>
