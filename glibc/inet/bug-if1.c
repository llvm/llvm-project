/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <net/if.h>


static int
do_test (void)
{
  char buf[IF_NAMESIZE];
  /* Index 0 is always invalid (see RFC 3493).  */
  char *cp = if_indextoname (0, buf);
  if (cp != NULL)
    {
      printf ("invalid index returned result \"%s\"\n", cp);
      return 1;
    }
  else if (errno != ENXIO)
    {
      int err = errno;
      char errbuf1[256];
      char errbuf2[256];

      printf ("errno = %d (%s), expected %d (%s)\n",
	      err, strerror_r (err, errbuf1, sizeof (errbuf1)),
	      ENXIO, strerror_r (ENXIO, errbuf2, sizeof (errbuf2)));
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
