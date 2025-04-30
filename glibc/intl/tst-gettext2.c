/* Test of the gettext functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de> and
   Andreas Jaeger <aj@suse.de>, 2000.

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
#include <libintl.h>
#include <stdlib.h>
#include <stdio.h>

#define N_(msgid) msgid

struct data_t
{
  const char *selection;
  const char *description;
};

int data_cnt = 2;
struct data_t strings[] =
{
  { "String1", N_("First string for testing.") },
  { "String2", N_("Another string for testing.") }
};

const int lang_cnt = 3;
const char *lang[] = {"lang1", "lang2", "lang3"};

static int
do_test (void)
{
  int i;

  /* Clean up environment.  */
  unsetenv ("LANGUAGE");
  unsetenv ("LC_ALL");
  unsetenv ("LC_MESSAGES");
  unsetenv ("LC_CTYPE");
  unsetenv ("LANG");
  unsetenv ("OUTPUT_CHARSET");

  textdomain ("tstlang");

  for (i = 0; i < lang_cnt; ++i)
    {
      int j;

      if (setlocale (LC_ALL, lang[i]) == NULL)
	setlocale (LC_ALL, "C");
      bindtextdomain ("tstlang", OBJPFX "domaindir");

      for (j = 0; j < data_cnt; ++j)
	printf ("%s - %s\n", strings[j].selection,
		gettext (strings[j].description));
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
