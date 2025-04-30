/* Functions for recorded errors, warnings, and verbose messages.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <error.h>
#include <errno.h>
#include <locale.h>

#include "record-status.h"

/* Warnings recorded by record_warnings.  */
int recorded_warning_count;

/* Errors recorded by record_errors.  */
int recorded_error_count;

/* If not zero suppress warnings and information messages.  */
int be_quiet;

/* If not zero give a lot more messages.  */
int verbose;

/* Warnings which can be disabled:  */
/* By default we check the character map for ASCII compatibility.  */
bool warn_ascii = true;
/* By default we check that the international currency symbol matches a
   known country code.  */
bool warn_int_curr_symbol = true;

/* Alter the current locale to match the locale configured by the
   user, and return the previous saved state.  */
struct locale_state
push_locale (void)
{
  int saved_errno;
  const char *orig;
  char *copy = NULL;

  saved_errno = errno;

  orig = setlocale (LC_CTYPE, NULL);
  if (orig == NULL)
    error (0, 0, "failed to read locale!");

  if (setlocale (LC_CTYPE, "") == NULL)
    error (0, 0, "failed to set locale!");

  errno = saved_errno;

  if (orig != NULL)
    copy = strdup (orig);

  /* We will return either a valid locale or NULL if we failed
     to save the locale.  */
  return (struct locale_state) { .cur_locale = copy };
}

/* Use the saved state to restore the locale.  */
void
pop_locale (struct locale_state ls)
{
  const char *set = NULL;
  /* We might have failed to save the locale, so only attempt to
     restore a validly saved non-NULL locale.  */
  if (ls.cur_locale != NULL)
    {
      set = setlocale (LC_CTYPE, ls.cur_locale);
      if (set == NULL)
	error (0, 0, "failed to restore %s locale!", ls.cur_locale);

      free (ls.cur_locale);
    }
}

/* Wrapper to print verbose informative messages.
   Verbose messages are only printed if --verbose
   is in effect and --quiet is not.  */
void
__attribute__ ((__format__ (__printf__, 2, 3), nonnull (1, 2), unused))
record_verbose (FILE *stream, const char *format, ...)
{
  char *str;
  va_list arg;

  if (!verbose)
    return;

  if (!be_quiet)
    {
      struct locale_state ls;
      int ret;

      va_start (arg, format);
      ls = push_locale ();

      ret = vasprintf (&str, format, arg);
      if (ret == -1)
	abort ();

      pop_locale (ls);
      va_end (arg);

      fprintf (stream, "[verbose] %s\n", str);

      free (str);
    }
}

/* Wrapper to print warning messages.  We keep track of how
   many were called because this effects our exit code.
   Nothing is printed if --quiet is in effect, but warnings
   are always counted.  */
void
__attribute__ ((__format__ (__printf__, 1, 2), nonnull (1), unused))
record_warning (const char *format, ...)
{
  char *str;
  va_list arg;

  recorded_warning_count++;

  if (!be_quiet)
    {
      struct locale_state ls;
      int ret;

      va_start (arg, format);
      ls = push_locale ();

      ret = vasprintf (&str, format, arg);
      if (ret == -1)
	abort ();

      pop_locale (ls);
      va_end (arg);

      fprintf (stderr, "[warning] %s\n", str);

      free (str);
    }
}

/* Wrapper to print error messages.  We keep track of how
   many were called because this effects our exit code.
   Nothing is printed if --quiet is in effect, but errors
   are always counted, and fatal errors always exit the
   program.  */
void
__attribute__ ((__format__ (__printf__, 3, 4), nonnull (3), unused))
record_error (int status, int errnum, const char *format, ...)
{
  char *str;
  va_list arg;

  recorded_error_count++;

  /* The existing behaviour is that even if you use --quiet, a fatal
     error is always printed and terminates the process.  */
  if (!be_quiet || status != 0)
    {
      struct locale_state ls;
      int ret;

      va_start (arg, format);
      ls = push_locale ();

      ret = vasprintf (&str, format, arg);
      if (ret == -1)
        abort ();

      pop_locale (ls);
      va_end (arg);

      error (status, errnum, "[error] %s", str);

      free (str);
    }
}
/* ... likewise for error_at_line.  */
void
__attribute__ ((__format__ (__printf__, 5, 6), nonnull (3, 5), unused))
record_error_at_line (int status, int errnum, const char *filename,
		      unsigned int linenum, const char *format, ...)
{
  char *str;
  va_list arg;

  recorded_error_count++;

  /* The existing behaviour is that even if you use --quiet, a fatal
     error is always printed and terminates the process.  */
  if (!be_quiet || status != 0)
    {
      struct locale_state ls;
      int ret;

      va_start (arg, format);
      ls = push_locale ();

      ret = vasprintf (&str, format, arg);
      if (ret == -1)
        abort ();

      pop_locale (ls);
      va_end (arg);

      error_at_line (status, errnum, filename, linenum, "[error] %s", str);

      free (str);
    }
}
