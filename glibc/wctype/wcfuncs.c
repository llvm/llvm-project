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

#include <ctype.h>
#include <wctype.h>
#include <locale/localeinfo.h>

#include "wchar-lookup.h"

/* Provide real-function versions of all the wctype macros.  */

#define	func(name, type)						      \
  extern int __isw##name (wint_t __wc);					      \
  int									      \
  __isw##name (wint_t wc)						      \
  {									      \
    if (isascii (wc))							      \
      return is##name ((int) wc);					      \
    size_t i = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_CLASS_OFFSET) + type;    \
    const char *desc = _NL_CURRENT (LC_CTYPE, i);			      \
    return wctype_table_lookup (desc, wc);				      \
  }									      \
  weak_alias (__isw##name, isw##name)

#undef iswalnum
func (alnum, __ISwalnum)
libc_hidden_def (__iswalnum)
libc_hidden_weak (iswalnum)
#undef iswalpha
func (alpha, __ISwalpha)
libc_hidden_weak (iswalpha)
#undef iswblank
func (blank, __ISwblank)
#undef iswcntrl
func (cntrl, __ISwcntrl)
#undef iswdigit
func (digit, __ISwdigit)
libc_hidden_weak (iswdigit)
#undef iswlower
func (lower, __ISwlower)
libc_hidden_def (__iswlower)
libc_hidden_weak (iswlower)
#undef iswgraph
func (graph, __ISwgraph)
#undef iswprint
func (print, __ISwprint)
#undef iswpunct
func (punct, __ISwpunct)
#undef iswspace
func (space, __ISwspace)
libc_hidden_weak (iswspace)
#undef iswupper
func (upper, __ISwupper)
#undef iswxdigit
func (xdigit, __ISwxdigit)
libc_hidden_weak (iswxdigit)

#undef towlower
wint_t
__towlower (wint_t wc)
{
  size_t i = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_MAP_OFFSET) + __TOW_tolower;
  const char *desc = _NL_CURRENT (LC_CTYPE, i);
  return wctrans_table_lookup (desc, wc);
}
libc_hidden_def (__towlower)
weak_alias (__towlower, towlower)
libc_hidden_weak (towlower)

#undef towupper
wint_t
__towupper (wint_t wc)
{
  size_t i = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_MAP_OFFSET) + __TOW_toupper;
  const char *desc = _NL_CURRENT (LC_CTYPE, i);
  return wctrans_table_lookup (desc, wc);
}
libc_hidden_def (__towupper)
weak_alias (__towupper, towupper)
libc_hidden_weak (towupper)
