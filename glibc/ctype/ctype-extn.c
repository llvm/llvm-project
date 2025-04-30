/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#define	__NO_CTYPE
#include <ctype.h>

#define __ctype_tolower \
  ((int32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TOLOWER) + 128)
#define __ctype_toupper \
  ((int32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TOUPPER) + 128)

/* Real function versions of the non-ANSI ctype functions.  */

int
_tolower (int c)
{
  return __ctype_tolower[c];
}
int
_toupper (int c)
{
  return __ctype_toupper[c];
}

int
toascii (int c)
{
  return __toascii (c);
}
weak_alias (toascii, __toascii_l)

int
isascii (int c)
{
  return __isascii (c);
}
weak_alias (isascii, __isascii_l)
