/* Error-checking wrapper for asprintf.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/support.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>

char *
xasprintf (const char *format, ...)
{
  va_list ap;
  va_start (ap, format);
  char *result;
  if (vasprintf (&result, format, ap) < 0)
    FAIL_EXIT1 ("asprintf: %m");
  va_end (ap);
  return result;
}
