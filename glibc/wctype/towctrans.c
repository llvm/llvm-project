/* Map wide character using given mapping.
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
#include "wchar-lookup.h"

wint_t
__towctrans (wint_t wc, wctrans_t desc)
{
  /* If the user passes in an invalid DESC valid (the one returned from
     `wctrans' in case of an error) simply return the value.  */
  if (desc == (wctrans_t) 0)
    return wc;

  return wctrans_table_lookup ((const char *) desc, wc);
}
libc_hidden_def (__towctrans)
weak_alias (__towctrans, towctrans)
