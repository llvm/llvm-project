/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <wcsmbs/wcsmbsload.h>


extern mbstate_t __wctomb_state attribute_hidden; /* Defined in wctomb.c.  */


int
__wctomb_chk (char *s, wchar_t wchar, size_t buflen)
{
  /* We do not have to implement the full wctomb semantics since we
     know that S cannot be NULL when we come here.  */
  if (buflen < MB_CUR_MAX)
    __chk_fail ();

  return __wcrtomb (s, wchar, &__wctomb_state);
}
