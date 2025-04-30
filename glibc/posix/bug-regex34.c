/* Test re_search with multi-byte characters in UTF-8.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#define _GNU_SOURCE 1
#include <stdio.h>
#include <string.h>
#include <locale.h>
#include <regex.h>

static int
do_test (void)
{
  struct re_pattern_buffer r;
  /* ကျွန်ုပ်x */
  const char *s = "\xe1\x80\x80\xe1\x80\xbb\xe1\x80\xbd\xe1\x80\x94\xe1\x80\xba\xe1\x80\xaf\xe1\x80\x95\xe1\x80\xbax";

  if (setlocale (LC_ALL, "en_US.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }
  memset (&r, 0, sizeof (r));

  re_compile_pattern ("[^x]x", 5, &r);
  /* This was triggering a buffer overflow.  */
  re_search (&r, s, strlen (s), 0, strlen (s), 0);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
