/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <wctype.h>
#include "../locale/localeinfo.h"

wctrans_t
__wctrans_l (const char *property, locale_t locale)
{
  const char *names;
  size_t cnt;
  size_t i;

  names = locale->__locales[LC_CTYPE]->values[_NL_ITEM_INDEX (_NL_CTYPE_MAP_NAMES)].string;
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

  i = locale->__locales[LC_CTYPE]->values[_NL_ITEM_INDEX (_NL_CTYPE_MAP_OFFSET)].word + cnt;
  return (wctrans_t) locale->__locales[LC_CTYPE]->values[i].string;
}
weak_alias (__wctrans_l, wctrans_l)
