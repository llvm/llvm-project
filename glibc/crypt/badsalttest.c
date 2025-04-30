/* Test program for bad DES salt detection in crypt.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <unistd.h>
#include <sys/mman.h>
#include <crypt.h>

static const char *tests[][2] =
  {
    { "no salt", "" },
    { "single char", "/" },
    { "first char bad", "!x" },
    { "second char bad", "Z%" },
    { "both chars bad", ":@" },
    { "un$upported algorithm", "$2$" },
    { "unsupported_algorithm", "_1" },
    { "end of page", NULL }
  };

static int
do_test (void)
{
  int result = 0;
  struct crypt_data cd;
  size_t n = sizeof (tests) / sizeof (*tests);
  size_t pagesize = (size_t) sysconf (_SC_PAGESIZE);
  char *page;

  /* Check that crypt won't look at the second character if the first
     one is invalid.  */
  page = mmap (NULL, pagesize * 2, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (page == MAP_FAILED)
    {
      perror ("mmap");
      n--;
    }
  else
    {
      if (mmap (page + pagesize, pagesize, 0,
		MAP_PRIVATE | MAP_ANON | MAP_FIXED,
		-1, 0) != page + pagesize)
	perror ("mmap 2");
      page[pagesize - 1] = '*';
      tests[n - 1][1] = &page[pagesize - 1];
    }

  /* Mark cd as initialized before first call to crypt_r.  */
  cd.initialized = 0;

  for (size_t i = 0; i < n; i++)
    {
      if (crypt (tests[i][0], tests[i][1]))
	{
	  result++;
	  printf ("%s: crypt returned non-NULL with salt \"%s\"\n",
		  tests[i][0], tests[i][1]);
	}

      if (crypt_r (tests[i][0], tests[i][1], &cd))
	{
	  result++;
	  printf ("%s: crypt_r returned non-NULL with salt \"%s\"\n",
		  tests[i][0], tests[i][1]);
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
