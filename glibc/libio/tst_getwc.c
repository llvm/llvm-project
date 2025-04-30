/* Simple test of getwc in the C locale.
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

#include <stdio.h>
#include <wchar.h>


int
main (void)
{
  wchar_t buf[100];
  size_t n = 0;

  wmemset (buf, L'\0', sizeof (buf) / sizeof (buf[0]));
  while (! feof (stdin) && n < sizeof (buf) - 1)
    {
      buf[n] = getwc (stdin);
      if (buf[n] == WEOF)
	break;
      ++n;
    }
  buf[n] = L'\0';

  return wcscmp (buf, L"This is a test of getwc\n") != 0;
}
