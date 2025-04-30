/* Dump the character classes and character maps of a locale to a bunch
   of individual files which can be processed with diff, sed etc.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <haible@clisp.cons.org>, 2000.

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

/* Usage example:
     $ dump-ctype de_DE.UTF-8
 */

#include <stdio.h>
#include <stdlib.h>
#include <wctype.h>
#include <locale.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

static const char *program_name = "dump-ctype";
static const char *locale;

static const char *class_names[] =
  {
    "alnum", "alpha", "blank", "cntrl", "digit", "graph", "lower",
    "print", "punct", "space", "upper", "xdigit"
  };

static const char *map_names[] =
  {
    "tolower", "toupper", "totitle"
  };

static void dump_class (const char *class_name)
{
  wctype_t class;
  FILE *f;
  unsigned int ch;

  class = wctype (class_name);
  if (class == (wctype_t) 0)
    {
      fprintf (stderr, "%s %s: noexistent class %s\n", program_name,
	       locale, class_name);
      return;
    }

  f = fopen (class_name, "w");
  if (f == NULL)
    {
      fprintf (stderr, "%s %s: cannot open file %s/%s\n", program_name,
	       locale, locale, class_name);
      exit (1);
    }

  for (ch = 0; ch < 0x10000; ch++)
    if (iswctype (ch, class))
      fprintf (f, "0x%04X\n", ch);

  if (ferror (f) || fclose (f))
    {
      fprintf (stderr, "%s %s: I/O error on file %s/%s\n", program_name,
	       locale, locale, class_name);
      exit (1);
    }
}

static void dump_map (const char *map_name)
{
  wctrans_t map;
  FILE *f;
  unsigned int ch;

  map = wctrans (map_name);
  if (map == (wctrans_t) 0)
    {
      fprintf (stderr, "%s %s: noexistent map %s\n", program_name,
	       locale, map_name);
      return;
    }

  f = fopen (map_name, "w");
  if (f == NULL)
    {
      fprintf (stderr, "%s %s: cannot open file %s/%s\n", program_name,
	       locale, locale, map_name);
      exit (1);
    }

  for (ch = 0; ch < 0x10000; ch++)
    if (towctrans (ch, map) != ch)
      fprintf (f, "0x%04X\t0x%04X\n", ch, towctrans (ch, map));

  if (ferror (f) || fclose (f))
    {
      fprintf (stderr, "%s %s: I/O error on file %s/%s\n", program_name,
	       locale, locale, map_name);
      exit (1);
    }
}

int
main (int argc, char *argv[])
{
  size_t i;

  if (argc != 2)
    {
      fprintf (stderr, "Usage: dump-ctype locale\n");
      exit (1);
    }
  locale = argv[1];

  if (setlocale (LC_ALL, locale) == NULL)
    {
      fprintf (stderr, "%s: setlocale cannot switch to locale %s\n",
	       program_name, locale);
      exit (1);
    }

  if (mkdir (locale, 0777) < 0)
    {
      char buf[100];
      int save_errno = errno;

      sprintf (buf, "%s: cannot create directory %s", program_name, locale);
      errno = save_errno;
      perror (buf);
      exit (1);
    }

  if (chdir (locale) < 0)
    {
      char buf[100];
      int save_errno = errno;

      sprintf (buf, "%s: cannot chdir to %s", program_name, locale);
      errno = save_errno;
      perror (buf);
      exit (1);
    }

  for (i = 0; i < sizeof (class_names) / sizeof (class_names[0]); i++)
    dump_class (class_names[i]);

  for (i = 0; i < sizeof (map_names) / sizeof (map_names[0]); i++)
    dump_map (map_names[i]);

  return 0;
}
