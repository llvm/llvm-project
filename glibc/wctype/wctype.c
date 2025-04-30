/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1996.

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

#include <endian.h>
#include <string.h>
#include <wctype.h>
#include <locale/localeinfo.h>

wctype_t
__wctype (const char *property)
{
  const char *names;
  unsigned int result;
  size_t proplen = strlen (property);
  size_t i;

  names = _NL_CURRENT (LC_CTYPE, _NL_CTYPE_CLASS_NAMES);
  for (result = 0; ; result++)
    {
      size_t nameslen = strlen (names);

      if (proplen == nameslen && memcmp (property, names, proplen) == 0)
	break;

      names += nameslen + 1;
      if (names[0] == '\0')
	return 0;
    }

  i = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_CLASS_OFFSET) + result;
  return (wctype_t) _NL_CURRENT_DATA (LC_CTYPE)->values[i].string;
}
weak_alias (__wctype, wctype)
