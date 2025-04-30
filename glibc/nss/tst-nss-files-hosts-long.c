/* Test getent doesn't fail with long /etc/hosts lines (Bug 21915).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* This test runs getent twice to parse a /etc/hosts with a very
   long line. Prior to fixing this parser this would crash getent.  */

#include <stdlib.h>
#include <nss.h>
#include <support/check.h>

static int
do_test (void)
{
  int ret;

  /* Run getent to fetch the IPv4 address for host test4.
     This forces /etc/hosts to be parsed.  */
  ret = system("getent ahostsv4 test4");
  if (ret != 0)
    FAIL_EXIT1("ahostsv4 failed");

  /* Likewise for IPv6.  */
  ret = system("getent ahostsv6 test6");
  if (ret != 0)
    FAIL_EXIT1("ahostsv6 failed");

  exit (0);
}

#include <support/test-driver.c>
