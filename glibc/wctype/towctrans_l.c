/* Map wide character using given mapping and locale.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <wctype.h>

/* Define the lookup function.  */
#define USE_IN_EXTENDED_LOCALE_MODEL	1
#include "wchar-lookup.h"

wint_t
__towctrans_l (wint_t wc, wctrans_t desc, locale_t locale)
{
  /* If the user passes in an invalid DESC valid (the one returned from
     `__wctrans_l' in case of an error) simply return the value.  */
  if (desc == (wctrans_t) 0)
    return wc;

  return wctrans_table_lookup ((const char *) desc, wc);
}
weak_alias (__towctrans_l, towctrans_l)
