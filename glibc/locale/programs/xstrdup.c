/* xstrdup.c -- copy a string with out of memory checking
   Copyright (C) 1990-2021 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#if defined STDC_HEADERS || defined HAVE_STRING_H || _LIBC
# include <string.h>
#else
# include <strings.h>
#endif
void *xmalloc (size_t n) __THROW;
char *xstrdup (char *string) __THROW;

/* Return a newly allocated copy of STRING.  */

char *
xstrdup (char *string)
{
  return strcpy (xmalloc (strlen (string) + 1), string);
}
