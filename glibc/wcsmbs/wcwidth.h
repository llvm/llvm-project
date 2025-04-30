/* Internal header containing implementation of wcwidth() function.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1996.

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

#include <wchar.h>
#include <wctype.h>
#include "../wctype/wchar-lookup.h"
#include "../locale/localeinfo.h"

/* Table containing width information.  */
extern const char *__ctype32_width attribute_hidden;

static __inline int
internal_wcwidth (wchar_t wc)
{
  unsigned char res;

  /* The tables have been prepared in such a way that
     1. wc == L'\0' yields res = 0,
     2. !iswprint (wc) implies res = '\xff'.  */
  res = wcwidth_table_lookup (_NL_CURRENT (LC_CTYPE, _NL_CTYPE_WIDTH), wc);

  return res == (unsigned char) '\xff' ? -1 : (int) res;
}
