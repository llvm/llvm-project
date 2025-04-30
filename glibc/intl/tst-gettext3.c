/* Test that the gettext() results come out in the correct encoding for
   locales that differ only in their encoding.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <bruno@clisp.org>, 2001, 2005.

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

#include <libintl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
do_test (void)
{
  char *s;
  int result = 0;

  unsetenv ("LANGUAGE");
  unsetenv ("OUTPUT_CHARSET");
  textdomain ("codeset");
  bindtextdomain ("codeset", OBJPFX "domaindir");

  setlocale (LC_ALL, "de_DE.ISO-8859-1");

  /* Here we expect output in ISO-8859-1.  */
  s = gettext ("cheese");
  if (strcmp (s, "K\344se"))
    {
      printf ("call 1 returned: %s\n", s);
      result = 1;
    }

  setlocale (LC_ALL, "de_DE.UTF-8");

  /* Here we expect output in UTF-8.  */
  s = gettext ("cheese");
  if (strcmp (s, "K\303\244se"))
    {
      printf ("call 2 returned: %s\n", s);
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
