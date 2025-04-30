/* asprintf with out of memory checking
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <libintl.h>
#include <error.h>

char *
xasprintf (const char *format, ...)
{
  va_list ap;
  va_start (ap, format);
  char *result;
  if (vasprintf (&result, format, ap) < 0)
    error (EXIT_FAILURE, 0, _("memory exhausted"));
  va_end (ap);
  return result;
}
