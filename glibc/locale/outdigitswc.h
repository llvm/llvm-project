/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 2000.

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

#include <assert.h>
#include <langinfo.h>
#include "../locale/localeinfo.h"

/* Look up the value of the next multibyte character and return its numerical
   value if it is one of the digits known in the locale.  If *DECIDED is
   -1 this means it is not yet decided which form it is and we have to
   search through all available digits.  Otherwise we know which script
   the digits are from.  */
static inline wchar_t
outdigitwc_value (int n)
{
  assert (0 <= n && n <= 9);

  return _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_OUTDIGIT0_WC + n);
}
