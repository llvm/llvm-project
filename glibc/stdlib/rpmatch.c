/* Determine whether string value is affirmation or negative response
   according to current locale's data.
   This file is part of the GNU C Library.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.

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

#include <langinfo.h>
#include <stdlib.h>
#include <regex.h>


/* Match against one of the response patterns, compiling the pattern
   first if necessary.  */
static int
try (const char *response,
     const int tag, const int match, const int nomatch,
     const char **lastp, regex_t *re)
{
  const char *pattern = nl_langinfo (tag);
  if (pattern != *lastp)
    {
      /* The pattern has changed.  */
      if (*lastp != NULL)
        {
          /* Free the old compiled pattern.  */
          __regfree (re);
          *lastp = NULL;
        }
      /* Compile the pattern and cache it for future runs.  */
      if (__regcomp (re, pattern, REG_EXTENDED) != 0)
        return -1;
      *lastp = pattern;
    }

  /* Try the pattern.  */
  return __regexec (re, response, 0, NULL, 0) == 0 ? match : nomatch;
}

int
rpmatch (const char *response)
{
  /* We cache the response patterns and compiled regexps here.  */
  static const char *yesexpr, *noexpr;
  static regex_t yesre, nore;

  return (try (response, YESEXPR, 1, 0, &yesexpr, &yesre) ?:
	  try (response, NOEXPR, 0, -1, &noexpr, &nore));
}
