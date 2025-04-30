/* Test for ,ccs= handling in fopen.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2001.

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
#include <locale.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <sys/resource.h>
#include <support/support.h>
#include <support/xstdio.h>

static const char inputfile[] = "../iconvdata/testdata/ISO-8859-1";

static int
do_bz17916 (void)
{
  /* BZ #17916 -- check invalid large ccs= case.  */
  struct rlimit rl;
  getrlimit (RLIMIT_STACK, &rl);
  rl.rlim_cur = 1024 * 1024;
  setrlimit (RLIMIT_STACK, &rl);

  const size_t sz = 2 * 1024 * 1024;
  char *ccs = xmalloc (sz);
  strcpy (ccs, "r,ccs=");
  memset (ccs + 6, 'A', sz - 6 - 1);
  ccs[sz - 1] = '\0';

  FILE *fp = fopen (inputfile, ccs);
  if (fp != NULL)
    {
      printf ("unxpected success\n");
      return 1;
    }
  free (ccs);

  return 0;
}

static int
do_test (void)
{
  FILE *fp;

  mtrace ();

  xsetlocale (LC_ALL, "de_DE.UTF-8");

  fp = xfopen (inputfile, "r,ccs=ISO-8859-1");

  while (! feof_unlocked (fp))
    {
      wchar_t buf[200];

      if (fgetws_unlocked (buf, sizeof (buf) / sizeof (buf[0]), fp) == NULL)
	break;

      fputws (buf, stdout);
    }

  xfclose (fp);

  return do_bz17916 ();
}

#include <support/test-driver.c>
