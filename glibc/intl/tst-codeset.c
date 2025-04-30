/* Test of bind_textdomain_codeset.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <haible@clisp.cons.org>, 2001.

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
#include <support/check.h>

static int
do_test (void)
{
  unsetenv ("LANGUAGE");
  unsetenv ("OUTPUT_CHARSET");
  setlocale (LC_ALL, "de_DE.ISO-8859-1");
  textdomain ("codeset");
  bindtextdomain ("codeset", OBJPFX "domaindir");

  /* Here we expect output in ISO-8859-1.  */
  TEST_COMPARE_STRING (gettext ("cheese"), "K\344se");

  /* Here we expect output in UTF-8.  */
  bind_textdomain_codeset ("codeset", "UTF-8");
  TEST_COMPARE_STRING (gettext ("cheese"), "K\303\244se");

  /* `a with umlaut' is transliterated to `ae'.  */
  bind_textdomain_codeset ("codeset", "ASCII//TRANSLIT");
  TEST_COMPARE_STRING (gettext ("cheese"), "Kaese");

  /* Transliteration also works by default even if not set.  */
  bind_textdomain_codeset ("codeset", "ASCII");
  TEST_COMPARE_STRING (gettext ("cheese"), "Kaese");

  return 0;
}

#include <support/test-driver.c>
