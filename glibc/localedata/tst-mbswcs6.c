/* Test for invalid input to wcrtomb.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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
#include <stdio.h>
#include <string.h>
#include <wchar.h>


static int
do_test (const char *loc)
{
  char buf[100];
  size_t n;
  mbstate_t state;
  const char *nloc;
  int res;

  nloc = setlocale (LC_ALL, loc);
  if (nloc == NULL)
    {
      printf ("could not set locale \"%s\"\n", loc);
      return 1;
    }
  printf ("new locale: %s\n", nloc);

  memset (&state, '\0', sizeof (state));
  errno = 0;
  n = wcrtomb (buf, (wchar_t) -15l, &state);

  printf ("n = %zd, errno = %d (%s)\n", n, errno, strerror (errno));

  res = n != (size_t) -1 || errno != EILSEQ;
  if (res)
    puts ("*** FAIL");
  putchar ('\n');

  return res;
}


int
main (void)
{
  int res;

  res = do_test ("C");
  res |= do_test ("de_DE.ISO-8859-1");
  res |= do_test ("de_DE.UTF-8");
  res |= do_test ("en_US.ANSI_X3.4-1968");
  res |= do_test ("ja_JP.EUC-JP");
  res |= do_test ("hr_HR.ISO-8859-2");
  //res |= do_test ("ru_RU.KOI8-R");

  return res;
}
