/* Test for inet_network.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2000.

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

#include <stdio.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

struct
{
  const char *network;
  uint32_t number;
} tests [] =
{
  {"1.0.0.0", 0x1000000},
  {"1.0.0", 0x10000},
  {"1.0", 0x100},
  {"1", 0x1},
  {"192.168.0.0", 0xC0A80000},
  {"0", 0},
  {"0x0", 0},
  /* Now some invalid addresses.  */
  {"0x", INADDR_NONE},
  {"1 bar", INADDR_NONE}, /* Bug 15277.  */
  {"141.30.225.2800", INADDR_NONE},
  {"141.76.1.1.1", INADDR_NONE},
  {"141.76.1.11.", INADDR_NONE},
  {"1410", INADDR_NONE},
  {"1.1410", INADDR_NONE},
  {"1.1410.", INADDR_NONE},
  {"1.1410", INADDR_NONE},
  {"141.76.1111", INADDR_NONE},
  {"141.76.1111.", INADDR_NONE}
};


static int
do_test (void)
{
  int errors = 0;
  size_t i;
  uint32_t res;

  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      printf ("Testing: %s\n", tests[i].network);
      res = inet_network (tests[i].network);

      if (res != tests[i].number)
	{
	  ++errors;
	  printf ("Test failed for inet_network (\"%s\"):\n",
		  tests[i].network);
	  printf ("Expected return value %u (0x%x) but got %u (0x%x).\n",
		  tests[i].number, tests[i].number, res, res);
	}

    }

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
