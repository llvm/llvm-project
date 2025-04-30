/* Tests re_comp and re_exec.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Isamu Hasegawa <isamu@yamato.ibm.com>, 2002.

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

#define _REGEX_RE_COMP
#include <sys/types.h>
#include <mcheck.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  const char *err;
  size_t i;
  int ret = 0;

  mtrace ();

  for (i = 0; i < 100; ++i)
    {
      err = re_comp ("a t.st");
      if (err)
	{
	  printf ("re_comp failed: %s\n", err);
	  ret = 1;
	}

      if (! re_exec ("This is a test."))
	{
	  printf ("re_exec failed\n");
	  ret = 1;
	}
    }

  return ret;
}
