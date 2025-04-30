/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1999.

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

#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct
{
  const char *path;
  const char *file;
} tests [] =
{
  { "/dir/file", "file" },
  { "file", "file"},
  { "/dir/file///", "file" },
  { "../file", "file" },
  { "/", "/" },
  { NULL, "."},
  { "", "."}
};


static int
do_test (void)
{
  size_t i = 0;
  int errors = 0;
  char path[1024];
  char *file;

  for (i = 0; i < sizeof (tests) / sizeof (tests [0]); ++i)
    {
      if (tests [i].path == NULL)
	file = __xpg_basename (NULL);
      else
	{
	  strcpy (path, tests [i].path);
	  file = __xpg_basename (path);
	}
      if (strcmp (file, tests [i].file))
	{
	  printf ("Test with `%s' failed: Result is: `%s'.\n",
		  (tests [i].path == NULL ? "NULL" : tests [i].path), file);
	  errors = 1;
	}
    }

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
