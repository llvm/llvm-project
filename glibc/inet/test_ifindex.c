/* Test interface name <-> index conversions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <Philip.Blundell@pobox.com>.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/if.h>

static int
do_test (void)
{
  int failures = 0;
  struct if_nameindex *idx = if_nameindex (), *p;
  if (idx == NULL)
    {
      if (errno != ENOSYS)
	{
	  printf ("Couldn't get any interfaces.\n");
	  exit (1);
	}
      /* The function is simply not implemented.  */
      exit (0);
    }

  printf ("Idx            Name | Idx           Name\n");

  for (p = idx; p->if_index || p->if_name; ++p)
    {
      char buf[IFNAMSIZ];
      unsigned int ni;
      int result;
      printf ("%3d %15s | ", p->if_index, p->if_name);
      printf ("%3d", ni = if_nametoindex (p->if_name));
      printf ("%15s", if_indextoname (p->if_index, buf));
      result = (ni != p->if_index || (strcmp (buf, p->if_name)));
      if (ni == p->if_index)
	/* We have to make sure that this is not an alias with the
	   same interface number.  */
	if (p->if_index == if_nametoindex (buf))
	  result = 0;
      printf ("%10s", result ? "fail" : "okay");
      printf ("\n");
      failures += result;
    }
  if_freenameindex (idx);
  return failures ? 1 : 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
