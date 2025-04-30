/* Test NUL handling of mbsrtowcs.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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
#include <string.h>
#include <wchar.h>

static int
do_test (void)
{
  const unsigned char buf[] = { 'a', 'b', '\0', 'c', 'd', '\0', 'e' };
  wchar_t out[sizeof (buf)];
  memset (out, '\xff', sizeof (out));

  mbstate_t state;
  memset (&state, '\0', sizeof (state));

  const char *in = (const char *) buf;
  size_t n = mbsrtowcs (out, &in, sizeof (out) / sizeof (wchar_t), &state);

  int result = 0;
  if (n != 2)
    {
      printf ("n = %zu, expected 2\n", n);
      result = 1;
    }
  if (in != NULL)
    {
      printf ("in = %p, expected NULL\n", in);
      result = 1;
    }
  if (out[0] != L'a')
    {
      printf ("out[0] = L'%lc', expected L'a'\n", (wint_t) out[0]);
      result = 1;
    }
  if (out[1] != L'b')
    {
      printf ("out[1] = L'%lc', expected L'b'\n", (wint_t) out[1]);
      result = 1;
    }
  if (out[2] != L'\0')
    {
      printf ("out[2] = L'%lc', expected L'\\0'\n", (wint_t) out[2]);
      result = 1;
    }
  return result;
}

#include <support/test-driver.c>
