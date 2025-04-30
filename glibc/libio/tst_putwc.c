/* Simple test of putwc in the C locale.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

static const char outname[] = OBJPFX "tst_putwc.temp";

/* Prototype for our test function.  */
int do_test (void);
#define TEST_FUNCTION do_test ()

/* This defines the `main' function and some more.  */
#include <test-skeleton.c>

int
do_test (void)
{
  const wchar_t str[] = L"This is a test of putwc\n";
  wchar_t buf[100];
  size_t n = 0;
  FILE *fp;
  int res = 0;

  add_temp_file (outname);

  fp = fopen (outname, "w+");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "cannot open temporary file");

  for (n = 0; str[n] != L'\0'; ++n)
    putwc (str[n], fp);

  /* First try reading after rewinding.  */
  rewind (fp);

  wmemset (buf, L'\0', sizeof (buf) / sizeof (buf[0]));
  n = 0;
  while (! feof (fp) && n < sizeof (buf) - 1)
    {
      buf[n] = getwc (fp);
      if (buf[n] == WEOF)
	break;
      ++n;
    }
  buf[n] = L'\0';

  if (wcscmp (buf, L"This is a test of putwc\n") != 0)
    {
      puts ("first comparison failed");
      res = 1;
    }

  /* Now close the file, open it again, and read again.  */
  if (fclose (fp) != 0)
    {
      printf ("failure during fclose: %m\n");
      res = 1;
    }

  fp = fopen (outname, "r");
  if (fp == NULL)
    {
      printf ("cannot reopen file: %m\n");
      return 1;
    }

  /* We can remove the file now.  */
  remove (outname);

  wmemset (buf, L'\0', sizeof (buf) / sizeof (buf[0]));
  n = 0;
  while (! feof (fp) && n < sizeof (buf) - 1)
    {
      buf[n] = getwc (fp);
      if (buf[n] == WEOF)
	break;
      ++n;
    }
  buf[n] = L'\0';

  if (wcscmp (buf, L"This is a test of putwc\n") != 0)
    {
      puts ("second comparison failed");
      res = 1;
    }

  if (fclose (fp) != 0)
    {
      printf ("failure during fclose: %m\n");
      res = 1;
    }

  /* Next test: write a bit more than a few bytes.  */
  fp = fopen (outname, "w");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "cannot open temporary file");

  for (n = 0; n < 4098; ++n)
    putwc (n & 255, fp);

  fclose (fp);

  return res;
}
