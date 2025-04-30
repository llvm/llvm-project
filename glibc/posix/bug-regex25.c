/* Test re_search in multibyte locale other than UTF-8.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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

#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <string.h>

const char *str1 = "\xa3\xd8\xa3\xc9\xa3\xc9";
const char *str2 = "\xa3\xd8\xa3\xc9";

int
main (void)
{
  setlocale (LC_ALL, "ja_JP.eucJP");

  re_set_syntax (RE_SYNTAX_SED);

  struct re_pattern_buffer re;
  memset (&re, 0, sizeof (re));

  struct re_registers regs;
  memset (&regs, 0, sizeof (regs));

  re_compile_pattern ("$", 1, &re);

  int ret = 0, r = re_search (&re, str1, 4, 0, 4, &regs);
  if (r != 4)
    {
      printf ("First re_search returned %d\n", r);
      ret = 1;
    }
  r = re_search (&re, str2, 4, 0, 4, &regs);
  if (r != 4)
    {
      printf ("Second re_search returned %d\n", r);
      ret = 1;
    }
  return ret;
}
