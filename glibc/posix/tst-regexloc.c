/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <regex.h>
#include <locale.h>
#include <stdio.h>

static int
do_test (void)
{
  regex_t re;
  regmatch_t mat[1];
  int res = 1;

  if (setlocale (LC_ALL, "de_DE.ISO-8859-1") == NULL)
    puts ("cannot set locale");
  /* Range expressions in non-POSIX locales are unspecified, but
     for now in glibc we maintain lowercase/uppercase distinction
     in our collation element order (but not in collation weights
     which means strcoll_l still collates as expected).  */
  else if (regcomp (&re, "[a-f]*", 0) != REG_NOERROR)
    puts ("cannot compile expression \"[a-f]*\"");
  else if (regexec (&re, "abcdefCDEF", 1, mat, 0) == REG_NOMATCH)
    puts ("no match");
  else
    {
      printf ("match from %d to %d\n", mat[0].rm_so, mat[0].rm_eo);
      res = mat[0].rm_so != 0 || mat[0].rm_eo != 6;
    }

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
