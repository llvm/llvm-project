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

#include <inttypes.h>
#include <string.h>
#include <wctype.h>
#include "../locale/localeinfo.h"

wctrans_t
__wctrans (const char *property)
{
  const char *names;
  size_t cnt;
  size_t i;

  names = _NL_CURRENT (LC_CTYPE, _NL_CTYPE_MAP_NAMES);
  cnt = 0;
  while (names[0] != '\0')
    {
      if (strcmp (property, names) == 0)
	break;

      names = strchr (names, '\0') + 1;
      ++cnt;
    }

  if (names[0] == '\0')
    return 0;

  i = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_MAP_OFFSET) + cnt;
  return (wctrans_t) _NL_CURRENT_DATA (LC_CTYPE)->values[i].string;
}
weak_alias (__wctrans, wctrans)
