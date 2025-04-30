/* Test legacy IPv4 text-to-address function inet_aton.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <array_length.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static const struct tests
{
  const char *input;
  int valid;
  uint32_t result;
} tests[] =
{
  { "", 0, 0 },
  { "-1", 0, 0 },
  { "256", 1, 0x00000100 },
  { "256.", 0, 0 },
  { "255a", 0, 0 },
  { "256a", 0, 0 },
  { "0x100", 1, 0x00000100 },
  { "0200.0x123456", 1, 0x80123456 },
  { "0300.0x89123456.", 0 ,0 },
  { "0100.-0xffff0000", 0, 0 },
  { "0.0xffffff", 1, 0x00ffffff },
  { "0.0x1000000", 0, 0 },
  { "0377.16777215", 1, 0xffffffff },
  { "0377.16777216", 0, 0 },
  { "0x87.077777777", 1, 0x87ffffff },
  { "0x87.0100000000", 0, 0 },
  { "0.1.3", 1, 0x00010003 },
  { "0.256.3", 0, 0 },
  { "256.1.3", 0, 0 },
  { "0.1.0x10000", 0, 0 },
  { "0.1.0xffff", 1, 0x0001ffff },
  { "0.1a.3", 0, 0 },
  { "0.1.a3", 0, 0 },
  { "1.2.3.4", 1, 0x01020304 },
  { "0400.2.3.4", 0, 0 },
  { "1.0x100.3.4", 0, 0 },
  { "1.2.256.4", 0, 0 },
  { "1.2.3.0x100", 0, 0 },
  { "323543357756889", 0, 0 },
  { "10.1.2.3.4", 0, 0 },
  { "192.0.2.1", 1, 0xc0000201 },
  { "192.0.2.2\nX", 1, 0xc0000202 },
  { "192.0.2.3 Y", 1, 0xc0000203 },
  { "192.0.2.3Z", 0, 0 },
  { "192.000.002.010", 1, 0xc0000208 },
};


static int
do_test (void)
{
  int result = 0;
  size_t cnt;

  for (cnt = 0; cnt < array_length (tests); ++cnt)
    {
      struct in_addr addr;

      if ((int) inet_aton (tests[cnt].input, &addr) != tests[cnt].valid)
	{
	  if (tests[cnt].valid)
	    printf ("\"%s\" not seen as valid IP address\n", tests[cnt].input);
	  else
	    printf ("\"%s\" seen as valid IP address\n", tests[cnt].input);
	  result = 1;
	}
      else if (tests[cnt].valid && addr.s_addr != ntohl (tests[cnt].result))
	{
	  printf ("\"%s\" not converted correctly: is %08x, should be %08x\n",
		  tests[cnt].input, addr.s_addr, tests[cnt].result);
	  result = 1;
	}
    }

  return result;
}

#include <support/test-driver.c>
