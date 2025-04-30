/* Test regcomp with collating symbols in bracket expressions
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

#include <stdio.h>
#include <string.h>
#include <locale.h>
#include <regex.h>

static int
do_test (void)
{
  regex_t r;

  if (setlocale (LC_ALL, "cs_CZ.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  if (regcomp (&r, "[[.ch.]]", REG_NOSUB) != 0)
    {
      puts ("regcomp failed");
      return 1;
    }

  if (regexec (&r, "ch", 0, 0, 0) != 0)
    {
      puts ("regexec failed");
      return 1;
    }

  regfree (&r);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
