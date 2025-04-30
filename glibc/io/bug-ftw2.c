/* Test for ftw function searching in current directory.
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


int cnt;
int result;
int sawown;
int sawcur;


static int
callback (const char *fname, const struct stat *st, int flag)
{
  printf ("%d: \"%s\" -> ", ++cnt, fname);
  if (strcmp (fname, ".") == 0 && sawcur)
    {
      puts ("current directory reported twice");
      result = 1;
    }
  else if (strcmp (fname, "./bug-ftw2.c") == 0 && sawown)
    {
      puts ("source file reported twice");
      result = 1;
    }
  else if (fname[0] != '.')
    {
      puts ("missing '.' as first character");
      result = 1;
    }
  else if (fname[1] != '\0' && fname[1] != '/')
    {
      puts ("no '/' in second position");
      result = 1;
    }
  else
    {
      puts ("OK");
      sawcur |= strcmp (fname, ".") == 0;
      sawown |= strcmp (fname, "./bug-ftw2.c") == 0;
    }

  return 0;
}


int
main (void)
{
  mtrace ();

  ftw (".", callback, 10);

  if (! sawcur)
    {
      puts ("current directory wasn't reported");
      result = 1;
    }

  if (! sawown)
    {
      puts ("source file wasn't reported");
      result = 1;
    }

  return result;
}
