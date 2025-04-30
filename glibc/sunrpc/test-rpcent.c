/* Test getrpcent and friends.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This is taken from nss/test-netdb.c and is intended to follow that
   test's model for everything.  This test is separate only because
   the <rpc/netdb.h> interfaces do not exist in configurations that
   omit sunrpc/ from the build.  */

#include <stdio.h>
#include <rpc/netdb.h>


static void
output_rpcent (const char *call, struct rpcent *rptr)
{
  char **pptr;

  if (rptr == NULL)
    printf ("Call: %s returned NULL\n", call);
  else
    {
      printf ("Call: %s, returned: r_name: %s, r_number: %d\n",
		call, rptr->r_name, rptr->r_number);
      for (pptr = rptr->r_aliases; *pptr != NULL; pptr++)
	printf ("  alias: %s\n", *pptr);
    }
}

static void
test_rpc (void)
{
  struct rpcent *rptr;

  rptr = getrpcbyname ("portmap");
  output_rpcent ("getrpcyname (\"portmap\")", rptr);

  rptr = getrpcbynumber (100000);
  output_rpcent ("getrpcbynumber (100000)", rptr);

  setrpcent (0);
  do
    {
      rptr = getrpcent ();
      output_rpcent ("getrpcent ()", rptr);
    }
  while (rptr != NULL);
  endrpcent ();
}

static int
do_test (void)
{
  test_rpc ();

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
