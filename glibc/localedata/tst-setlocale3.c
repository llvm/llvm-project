/* Regression test for setlocale invalid environment variable handling.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The result of setlocale may be overwritten by subsequent calls, so
   this wrapper makes a copy.  */
static char *
setlocale_copy (int category, const char *locale)
{
  const char *result = setlocale (category, locale);
  if (result == NULL)
    return NULL;
  return strdup (result);
}

static char *de_locale;

static void
setlocale_fail (const char *envstring)
{
  setenv ("LC_CTYPE", envstring, 1);
  if (setlocale (LC_CTYPE, "") != NULL)
    {
      printf ("unexpected setlocale success for \"%s\" locale\n", envstring);
      exit (1);
    }
  const char *newloc = setlocale (LC_CTYPE, NULL);
  if (strcmp (newloc, de_locale) != 0)
    {
      printf ("failed setlocale call \"%s\" changed locale to \"%s\"\n",
	      envstring, newloc);
      exit (1);
    }
}

static void
setlocale_success (const char *envstring)
{
  setenv ("LC_CTYPE", envstring, 1);
  char *newloc = setlocale_copy (LC_CTYPE, "");
  if (newloc == NULL)
    {
      printf ("setlocale for \"%s\": %m\n", envstring);
      exit (1);
    }
  if (strcmp (newloc, de_locale) == 0)
    {
      printf ("setlocale with LC_CTYPE=\"%s\" left locale at \"%s\"\n",
	      envstring, de_locale);
      exit (1);
    }
  if (setlocale (LC_CTYPE, de_locale) == NULL)
    {
      printf ("restoring locale \"%s\" with LC_CTYPE=\"%s\": %m\n",
	      de_locale, envstring);
      exit (1);
    }
  char *newloc2 = setlocale_copy (LC_CTYPE, newloc);
  if (newloc2 == NULL)
    {
      printf ("restoring locale \"%s\" following \"%s\": %m\n",
	      newloc, envstring);
      exit (1);
    }
  if (strcmp (newloc, newloc2) != 0)
    {
      printf ("representation of locale \"%s\" changed from \"%s\" to \"%s\"",
	      envstring, newloc, newloc2);
      exit (1);
    }
  free (newloc);
  free (newloc2);

  if (setlocale (LC_CTYPE, de_locale) == NULL)
    {
      printf ("restoring locale \"%s\" with LC_CTYPE=\"%s\": %m\n",
	      de_locale, envstring);
      exit (1);
    }
}

/* Checks that a known-good locale still works if LC_ALL contains a
   value which should be ignored.  */
static void
setlocale_ignore (const char *to_ignore)
{
  const char *fr_locale = "fr_FR.UTF-8";
  setenv ("LC_CTYPE", fr_locale, 1);
  char *expected_locale = setlocale_copy (LC_CTYPE, "");
  if (expected_locale == NULL)
    {
      printf ("setlocale with LC_CTYPE=\"%s\" failed: %m\n", fr_locale);
      exit (1);
    }
  if (setlocale (LC_CTYPE, de_locale) == NULL)
    {
      printf ("failed to restore locale: %m\n");
      exit (1);
    }
  unsetenv ("LC_CTYPE");

  setenv ("LC_ALL", to_ignore, 1);
  setenv ("LC_CTYPE", fr_locale, 1);
  const char *actual_locale = setlocale (LC_CTYPE, "");
  if (actual_locale == NULL)
    {
      printf ("setlocale with LC_ALL, LC_CTYPE=\"%s\" failed: %m\n",
	      fr_locale);
      exit (1);
    }
  if (strcmp (actual_locale, expected_locale) != 0)
    {
      printf ("setlocale under LC_ALL failed: got \"%s\", expected \"%s\"\n",
	      actual_locale, expected_locale);
      exit (1);
    }
  unsetenv ("LC_CTYPE");
  setlocale_success (fr_locale);
  unsetenv ("LC_ALL");
  free (expected_locale);
}

static int
do_test (void)
{
  /* The glibc test harness sets this environment variable
     uncondionally.  */
  unsetenv ("LC_ALL");

  de_locale = setlocale_copy (LC_CTYPE, "de_DE.UTF-8");
  if (de_locale == NULL)
    {
      printf ("setlocale (LC_CTYPE, \"de_DE.UTF-8\"): %m\n");
      return 1;
    }
  setlocale_success ("C");
  setlocale_success ("en_US.UTF-8");
  setlocale_success ("/en_US.UTF-8");
  setlocale_success ("//en_US.UTF-8");
  setlocale_ignore ("");

  setlocale_fail ("does-not-exist");
  setlocale_fail ("/");
  setlocale_fail ("/../localedata/en_US.UTF-8");
  setlocale_fail ("en_US.UTF-8/");
  setlocale_fail ("en_US.UTF-8/..");
  setlocale_fail ("en_US.UTF-8/../en_US.UTF-8");
  setlocale_fail ("../localedata/en_US.UTF-8");
  {
    size_t large_length = 1024;
    char *large_name = malloc (large_length + 1);
    if (large_name == NULL)
      {
	puts ("malloc failure");
	return 1;
      }
    memset (large_name, '/', large_length);
    const char *suffix = "en_US.UTF-8";
    strcpy (large_name + large_length - strlen (suffix), suffix);
    setlocale_fail (large_name);
    free (large_name);
  }
  {
    size_t huge_length = 64 * 1024 * 1024;
    char *huge_name = malloc (huge_length + 1);
    if (huge_name == NULL)
      {
	puts ("malloc failure");
	return 1;
      }
    memset (huge_name, 'X', huge_length);
    huge_name[huge_length] = '\0';
    /* Construct a composite locale specification. */
    const char *prefix = "LC_CTYPE=de_DE.UTF-8;LC_TIME=";
    memcpy (huge_name, prefix, strlen (prefix));
    setlocale_fail (huge_name);
    free (huge_name);
  }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
