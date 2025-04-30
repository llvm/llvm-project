/* Test system cache line sizes.
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <array_length.h>

static struct
{
  const char *name;
  int _SC_val;
} sc_options[] =
  {
#define N(name) { "_SC_"#name, _SC_##name }
    N (LEVEL1_ICACHE_LINESIZE),
    N (LEVEL1_DCACHE_LINESIZE),
    N (LEVEL2_CACHE_LINESIZE)
  };

static int
do_test (void)
{
  int result = EXIT_SUCCESS;

  for (int i = 0; i < array_length (sc_options); ++i)
    {
      long int scret = sysconf (sc_options[i]._SC_val);
      if (scret < 0)
	{
	  printf ("sysconf (%s) returned < 0 (%ld)\n",
		  sc_options[i].name, scret);
	  result = EXIT_FAILURE;
	}
      else
	printf ("sysconf (%s): %ld\n", sc_options[i].name, scret);
    }

  return result;
}

#include <support/test-driver.c>
