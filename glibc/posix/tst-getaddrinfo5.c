/* Test host lookup with double dots at the end, [BZ #16469].
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>

static int
test (void)
{
  static char host1[] = "localhost..";
  static char host2[] = "www.gnu.org..";
  static char *hosts[] = { host1, host2 };
  int i;
  int pass = 0;

  for (i = 0; i < sizeof (hosts) / sizeof (*hosts); i++)
    {
      char *host = hosts[i];
      size_t len = strlen (host);
      struct addrinfo *ai;

      /* If the name doesn't resolve with a single dot at the
	 end, skip it.  */
      host[len-1] = 0;
      if (getaddrinfo (host, NULL, NULL, &ai) != 0)
	{
	  printf ("resolving \"%s\" failed, skipping this hostname\n", host);
	  continue;
	}
      printf ("resolving \"%s\" worked, proceeding to test\n", host);
      freeaddrinfo (ai);

      /* If it resolved with a single dot, check that it doesn't with
	 a second trailing dot.  */
      host[len-1] = '.';
      if (getaddrinfo (host, NULL, NULL, &ai) == 0)
	{
	  printf ("resolving \"%s\" worked, test failed\n", host);
	  return 1;
	}
      printf ("resolving \"%s\" failed, test passed\n", host);
      pass = 1;
    }

  /* We want at least one successful name resolution for the test to
     succeed.  */
  return pass ? 0 : 2;
}

#define TEST_FUNCTION test ()
#include "../test-skeleton.c"
