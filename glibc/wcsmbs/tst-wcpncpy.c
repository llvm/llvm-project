/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2003.

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


static int
do_test (void)
{
  int result = 0;

  const wchar_t src[] = L"0";
  wchar_t dest[21];
  wmemset (dest, L'\0', 10);
  wchar_t *endp = wcpncpy (dest, src, 2);
  if (wcscmp (dest, src) != 0)
    {
      result = 1;
      puts ("L\"0\" string test failed");
    }
  if (endp != dest + 1)
    {
      result = 1;
      puts ("return value of L\"0\" string call incorrect");
    }

  const wchar_t src2[] = L"abc";
  endp = wcpncpy (dest, src2, 2);
  if (endp != dest + 2)
    {
      result = 1;
      puts ("return value of limited call incorrect");
    }

  const wchar_t src3[] = L"";
  endp = wcpncpy (dest, src3, 2);
  if (endp != dest)
    {
      result = 1;
      puts ("return value of empty string call incorrect");
    }

  const wchar_t src4[] = L"abcdefghijklmnopqrstuvwxyz";
  endp = wcpncpy (dest, src4, 2);
  if (endp != dest + 2)
    {
      result = 1;
      puts ("return value of long string call incorrect");
    }

  const wchar_t src5[] = L"ab";
  endp = wcpncpy (dest, src5, 20);
  if (endp != dest + 2)
    {
      result = 1;
      puts ("return value of large limit call incorrect");
    }

  return result;
}

#include <support/test-driver.c>
