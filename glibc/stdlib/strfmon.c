/* Formatting a monetary value according to the current locale.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>
   and Jochen Hein <Jochen.Hein@informatik.TU-Clausthal.de>, 1996.

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

#include <monetary.h>
#include <stdarg.h>
#include <locale/localeinfo.h>
#include <math_ldbl_opt.h>

ssize_t
__strfmon (char *s, size_t maxsize, const char *format, ...)
{
  va_list ap;

  va_start (ap, format);

  ssize_t res = __vstrfmon_l_internal (s, maxsize, _NL_CURRENT_LOCALE,
				       format, ap, 0);

  va_end (ap);

  return res;
}
ldbl_strong_alias (__strfmon, strfmon)
