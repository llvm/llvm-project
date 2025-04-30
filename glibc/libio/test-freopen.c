/* Test for freopen implementation.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>


int
main (int argc, char *argv[])
{
  FILE *fp;

  mtrace ();

  if (argc < 2)
    exit (1);

  fp = fopen (argv[1], "w");
  if (fp == NULL)
    {
      puts ("fopen failed: %m");
      exit (1);
    }

  fputs ("Hello world (mb)\n", fp);

  fp = freopen (argv[1], "a+", fp);
  if (fp == NULL)
    {
      puts ("freopen failed: %m");
      exit (1);
    }

  fputws (L"Hello world (wc)\n", fp);

  fclose (fp);

  return 0;
}
