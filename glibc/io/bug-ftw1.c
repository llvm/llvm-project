/* Test for ftw function searching in root directory.
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

#include <ftw.h>
#include <mcheck.h>
#include <stdio.h>
#include <string.h>


int result;
int cnt;
int sawroot;


static int
callback (const char *fname, const struct stat *st, int flag)
{
  if (++cnt >= 10)
    return 1;

  printf ("%d: \"%s\" -> ", cnt, fname);
  if (strcmp (fname, "/") == 0 && sawroot)
    {
      puts ("root directory reported twice");
      result = 1;
    }
  else if (fname[0] != '/')
    {
      puts ("missing '/' as first character");
      result = 1;
    }
  else if (fname[1] == '/')
    {
      puts ("double '/' at beginning");
      result = 1;
    }
  else
    {
      puts ("OK");
      sawroot |= strcmp (fname, "/") == 0;
    }

  return 0;
}


int
main (void)
{
  mtrace ();

  ftw ("/", callback, 10);

  if (! sawroot)
    {
      puts ("root directory wasn't reported");
      result = 1;
    }

  return result;
}
